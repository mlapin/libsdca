"""Softmax Collapse and Softmax Cancellation (Barron, Gnanapragasam and Tschannen, 2026) in
every loss, dtype and backend.

Collapse. Standard cross-entropy rounds 1 + tau to 1 once the margin passes the precision cliff
c_d = p_d ln 2 (5.5 bfloat16, 7.6 float16, 16.6 float32, 36.7 float64): the loss and the
target-logit gradient become exactly 0 while the competitor entries stay representable, so
the gradient no longer sums to zero. test_collapse sweeps the margin of rows of 2 and 4
classes (label on top, centered, offset) from negative values through the hinges' margin 1
and the cliff to where the exact loss and gradient underflow the output dtype, and, in its
own call, to margins near the dtype's max. Every row checks:

- the loss is positive wherever the exact loss rounds to a nonzero value of its dtype, and
  within BUDGET of it;
- the target entry (the label, the positives, the teacher's top class) is nonzero wherever
  the exact one rounds to nonzero and within ULPS of it; the other entries are within
  BUDGET;
- the row's gradient sums to zero within the rounding of the output dtype (ZERO_SUM).

Cancellation. Soft-target cross-entropy that forms q - p from two softmaxes has a floor where
q = p (2.2e-3 bfloat16, 2.6e-8 float32, 4.3e-17 float64). test_cancellation runs the paper's
Fig. 1b sweep (teacher logits z_T, student z_T + g d with |d|_inf = 1, g from 1 to 0) in one
call: distillation and proximal distillation are exactly 0 at g = 0 in every dtype and,
elsewhere, within the bounds above of the exact gradient of the represented inputs, whose
absolute allowance shrinks with the represented gap (Eq. 9 of the paper, evaluated in the
dtype, meets it). Cross-entropy of the probability targets softmax(z_T), rounded to the dtype
and so not exactly the teacher's, has the floor: its allowance is the work's rounding of the
larger of T q_j and u_j.

The references are conformance.references (mpmath) of the represented inputs: of labels,
masks, teacher logits and target masses (probability targets, label smoothing and class
weights). Backends: libsdca.loss_and_grad (NumPy), the CPU kernels and the composed
operations of libsdca.torch and libsdca.jax; the CUDA cells (Triton kernels and composed
operations) carry the cuda marker.

Tolerances (eps: the dtype's machine epsilon 2^(1 - p), u = eps / 2 its unit roundoff, eta
its smallest subnormal, tiny its smallest normal):

- ULPS (8, the documented accuracy of cross-entropy, docs/API.md, Known limits) and BUDGET
  (64, the documented bound of every loss, conformance/contract.py): |x^ - x| <= K (eps_out
  |x| + eps_work a) + eta_out, a the documented allowance of the entry: 0 for cross-entropy,
  the hinge margins' 1 + spread (over gamma for the smoothed gradients), the proximal
  gradient's 1-norm, distillation's (p + q) / T times min(1, |z - t|_inf / T);
- ZERO_SUM: |sum_j g^_j| <= C u_out sum_j |g^_j| + C eta_out. The exact gradient of every
  loss sums to zero (each is invariant to adding a constant to the logits). Each of the C
  stored entries is within u_out |g_j| (or eta_out / 2 for subnormals) of a value of the
  work, and a work sum of C - 1 of them (the target entry formed as minus the sum of the
  others, or each entry rounded alone) adds at most (C - 2) u_work sum_j |g_j|, with
  u_work <= u_out: at most C u_out sum_j |g_j| in all. A collapsed row misses it by
  sum_j |g^_j| / 2. Rows whose targets the work rounds add ULPS eps_work sum_j a_j.

Documented flushes (docs/API.md, Known limits): XLA flushes subnormal inputs and outputs, so
JAX results may be 0 where the exact value is below the output dtype's smallest normal; the
proximal losses return 0 below the work's smallest normal. ZERO_SUM then adds C times that
smallest normal. The rows near finfo.max keep the label on top and their scores at most
finfo.max apart, inside the documented range of every loss.
"""

from __future__ import annotations

import functools
import importlib.util
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import libsdca
import mpmath as mp
import named_losses
import numpy as np
import pytest
from conformance import references
from conformance.contract import BUDGET
from conformance.references import DISTILLATION, HINGES, PROXIMAL, Block, Loss, reference
from mpmath_types import MPReal
from storage_limits import LIMITS

DTYPES = ("bfloat16", "float16", "float32", "float64")
# (significand bits p, smallest normal, smallest subnormal, max)
FORMATS = {d: (lim.digits, lim.tiny, lim.tiny * lim.eps, lim.max) for d, lim in LIMITS.items()}
ULPS = 8
GAPS = (1.0, 1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 0.0)


def eps(dtype: str) -> float:
    return 2.0 ** (1 - FORMATS[dtype][0])


def cliff(dtype: str) -> float:
    """c_d = p_d ln 2: two-class margins beyond it round 1 + exp(-m) to 1."""
    return FORMATS[dtype][0] * math.log(2)


@functools.cache
def _numpy_dtype(dtype: str) -> Any:
    if dtype == "bfloat16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return np.dtype(dtype)


def represent(x: Any, dtype: str) -> np.ndarray:
    """x rounded to dtype (to nearest even), as float64."""
    return np.asarray(np.asarray(x, np.float64).astype(_numpy_dtype(dtype)), np.float64)


def rounds_to_zero(x: MPReal, dtype: str) -> bool:
    return abs(x) <= mp.mpf(FORMATS[dtype][2]) / 2


# ---------------------------------------------------------------------------
# Losses.


@dataclass(frozen=True)
class Config:
    """A loss of the sweep: its reference Loss, the kind of target and the options of
    cross-entropy (label smoothing and class weights, in PyTorch and JAX)."""

    label: str
    loss: Loss
    target: str = "labels"  # labels, mask, probabilities or teacher
    smoothing: float = 0.0
    weighted: bool = False

    @property
    def name(self) -> str:
        return self.loss.name

    @property
    def wide(self) -> bool:
        """Whether the loss needs rows of more than two classes (k = 2)."""
        return self.loss.k > 1

    @property
    def multilabel(self) -> bool:
        return self.target == "mask"

    @property
    def options(self) -> bool:
        return self.smoothing > 0 or self.weighted


CONFIGS = (
    Config("cross_entropy", Loss("cross_entropy")),
    Config("cross_entropy-probabilities", Loss("cross_entropy"), "probabilities"),
    Config("cross_entropy-smoothing", Loss("cross_entropy"), smoothing=0.1),
    Config("cross_entropy-weight", Loss("cross_entropy"), weighted=True),
    Config("topk_cross_entropy-k2", Loss("topk_cross_entropy", k=2)),
    Config("proximal_cross_entropy-a0.1", Loss("proximal_cross_entropy", alpha=0.1)),
    Config("multiclass_hinge", Loss("multiclass_hinge")),
    Config("multiclass_hinge-g0.5", Loss("multiclass_hinge", gamma=0.5)),
    Config("topk_hinge_alpha-k2", Loss("topk_hinge_alpha", k=2)),
    Config("topk_hinge_alpha-k2-g0.5", Loss("topk_hinge_alpha", k=2, gamma=0.5)),
    Config("topk_hinge_beta-k2", Loss("topk_hinge_beta", k=2)),
    Config("topk_hinge_beta-k2-g0.5", Loss("topk_hinge_beta", k=2, gamma=0.5)),
    Config("multilabel_cross_entropy", Loss("multilabel_cross_entropy"), "mask"),
    Config("multilabel_hinge", Loss("multilabel_hinge"), "mask"),
    Config("multilabel_hinge-g0.5", Loss("multilabel_hinge", gamma=0.5), "mask"),
    Config("distillation-T1", Loss("distillation"), "teacher"),
    Config("distillation-T4", Loss("distillation", temperature=4.0), "teacher"),
    Config(
        "proximal_distillation-T1-a0.1",
        Loss("proximal_distillation", alpha=0.1),
        "teacher",
    ),
)
CANCELLATION = (
    Config("cross_entropy-probabilities", Loss("cross_entropy"), "probabilities"),
    Config("distillation-T1", Loss("distillation"), "teacher"),
    Config("distillation-T4", Loss("distillation", temperature=4.0), "teacher"),
    Config(
        "proximal_distillation-T1-a0.1",
        Loss("proximal_distillation", alpha=0.1),
        "teacher",
    ),
)
WEIGHTS = (2.0, 0.5, 1.5, 0.25)


# ---------------------------------------------------------------------------
# Rows.


@dataclass(frozen=True)
class Rows:
    """A batch of rows of one width: represented logits, targets (labels, masks,
    probabilities or teacher logits) and the tag of each row."""

    family: str
    logits: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    teacher: np.ndarray
    probabilities: np.ndarray
    tags: tuple[str, ...]

    @property
    def classes(self) -> int:
        return self.logits.shape[1]


def margins(dtype: str) -> list[float]:
    """Margins from below the hinges' 1 through the cliff to the underflow of the loss and
    of the gradient of the output dtypes (and of float32 work), and on to 1e20."""
    c = cliff(dtype)
    out = [-20.0, -2.0, -0.5, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 4.0]
    out += [c * f for f in (0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0)]
    for limit_dtype in {dtype, "float32"} | ({"float64"} if dtype == "float64" else set()):
        _, tiny, eta, _ = FORMATS[limit_dtype]
        for x in (-math.log(tiny), -math.log(eta)):
            out += [x * f for f in (0.95, 1.0, 1.05)] + [x + 5]
    out += [1e3, 1e4, 1e6, 1e10, 1e20]
    big = FORMATS[dtype][3]
    kept = represent([m for m in out if m < big / 8], dtype)
    return sorted(set(kept.tolist()))


def extreme_margins(dtype: str) -> list[float]:
    big = FORMATS[dtype][3]
    out = [big / 8, big / 4, big / 2, big]
    out += [1e30, 1e38, 1e100, 1e300]
    return sorted(set(represent([m for m in out if m <= big], dtype).tolist()))


def _layouts(config: Config, m: float) -> Iterator[tuple[str, list[float], list[bool]]]:
    """(family, logits, positives) of margin m: the label at 0 above competitors at -m, on
    two classes (pair), on four (four: a tie at -m and -(m + 1); two positives for the
    multilabel losses), centered at zero (centered) and offset by 100 (offset)."""
    if not config.wide:
        yield "pair", [0.0, -m], [True, False]
        yield "centered", [m / 2, -m / 2], [True, False]
        yield "offset", [100.0, 100.0 - m], [True, False]
    if config.multilabel:
        yield "four", [0.0, 0.0, -m, -(m + 1)], [True, True, False, False]
    else:
        yield "four", [0.0, -m, -m, -(m + 1)], [True, False, False, False]


def _softmax(values: Sequence[float] | np.ndarray) -> list[float]:
    with mp.workdps(60):
        top = max(values)
        e = [mp.exp(mp.mpf(v) - top) for v in values]
        total = mp.fsum(e)
        return [float(x / total) for x in e]


def collapse_rows(config: Config, dtype: str, extreme: bool) -> list[Rows]:
    """The margin sweep of config in dtype, one Rows per width; teacher logits are half the
    student's, probability targets one-hot and, in the soft rows, softmax(z / 2)."""
    tiny = FORMATS[dtype][1]
    big = FORMATS[dtype][3]
    widths: dict[int, list[tuple[np.ndarray, list[bool], np.ndarray, np.ndarray, str]]] = {}
    for m in extreme_margins(dtype) if extreme else margins(dtype):
        for family, values, positive in _layouts(config, m):
            z = represent(np.clip(values, -big, big), dtype)
            teacher = represent(z / 2, dtype)
            onehot = np.array(positive, np.float64)
            rows = [(z, positive, teacher, onehot, f"{family} m={m:.6g}")]
            if config.target == "probabilities":
                soft = represent(_softmax(z / 2), dtype)
                soft[soft < tiny] = 0.0
                rows.append((z, positive, teacher, soft, f"{family}-soft m={m:.6g}"))
            widths.setdefault(len(values), []).extend(rows)
    return [
        Rows(
            f"{'extreme' if extreme else 'sweep'}{width}",
            np.array([r[0] for r in rows]),
            np.zeros(len(rows), np.int64),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]),
            np.array([r[3] for r in rows]),
            tuple(r[4] for r in rows),
        )
        for width, rows in widths.items()
    ]


@functools.cache
def _sweep_inputs(rows: int = 4, classes: int = 53) -> tuple[np.ndarray, np.ndarray]:
    """Fig. 1b's Gaussian teacher logits and unit directions |d|_inf = 1."""
    rng = np.random.default_rng(0)
    teacher = rng.standard_normal((rows, classes))
    direction = rng.standard_normal((rows, classes))
    return teacher, direction / np.abs(direction).max(-1, keepdims=True)


def cancellation_rows(dtype: str) -> tuple[Rows, list[float]]:
    """Students z_T + g d of every gap g and teachers z_T, rounded to dtype, and the gap of
    each row; the probability targets are softmax(z_T) rounded to dtype (zero below its
    smallest normal)."""
    zt, d = _sweep_inputs()
    teacher = np.tile(represent(zt, dtype), (len(GAPS), 1))
    student = represent(np.concatenate([zt + g * d for g in GAPS]), dtype)
    p = represent([_softmax(row) for row in zt], dtype)
    p[p < FORMATS[dtype][1]] = 0.0
    n = len(student)
    gaps = [g for g in GAPS for _ in zt]
    rows = Rows(
        "cancellation",
        student,
        np.zeros(n, np.int64),
        np.zeros(student.shape, bool),
        teacher,
        np.tile(p, (len(GAPS), 1)),
        tuple(f"g={g:g} row {i % len(zt)}" for i, g in enumerate(gaps)),
    )
    return rows, gaps


# ---------------------------------------------------------------------------
# References.


@dataclass(frozen=True)
class Exact:
    """A row's exact loss, gradient entries, documented allowance per entry (in work eps
    units) and target entries; kinked: a nonsmooth hinge at a kink (several subgradients);
    strict: the represented inputs determine the target entries to relative accuracy (not
    so for probability targets that the work rounds: smoothed or not one-hot)."""

    value: MPReal
    gradient: tuple[MPReal, ...]
    allowance: tuple[float, ...]
    value_allowance: float
    targets: tuple[int, ...]
    kinked: bool = False
    strict: bool = True


def _blocks(config: Config, rows: Rows, i: int) -> tuple[tuple[Block, ...], list[int]]:
    """Row i as blocks of equal (logit, positive, teacher) entries, and each entry's block."""
    keys: dict[tuple[float, bool, float], int] = {}
    index = []
    positive = rows.mask[i] if config.multilabel else np.arange(rows.classes) == rows.labels[i]
    for j in range(rows.classes):
        teacher = float(rows.teacher[i, j]) if config.target == "teacher" else 0.0
        key = (float(rows.logits[i, j]), bool(positive[j]), teacher)
        index.append(keys.setdefault(key, len(keys)))
    counts = [index.count(b) for b in range(len(keys))]
    row = tuple(Block(k[0], n, k[1], k[2]) for k, n in zip(keys, counts, strict=True))
    return row, index


def _probability_reference(config: Config, rows: Rows, i: int) -> Exact:
    c = rows.classes
    if config.target == "probabilities":
        t = rows.probabilities[i : i + 1]
    else:
        t = np.zeros((1, c))
        t[0, rows.labels[i]] = 1.0
    u = t * (1 - config.smoothing) + config.smoothing / c
    if config.weighted:
        u = u * np.array(WEIGHTS[:c])
    ((row, index),) = references.rows(rows.logits[i : i + 1], u)
    ref = reference(config.loss, row)
    gradient = tuple(ref.gradient[b] for b in index)
    onehot = bool(np.isin(t, (0.0, 1.0)).all() and t.sum() == 1)
    strict = onehot and config.smoothing == 0
    # T q_j - u_j of targets u the work rounds: the rounding of T = sum_j u_j.
    allowance = (0.0,) * c if strict else (float(u.sum()),) * c
    top = int(np.argmax(t[0]))
    return Exact(ref.value, gradient, allowance, 0.0, (top,), strict=strict)


def exact(config: Config, rows: Rows, i: int) -> Exact:
    """The reference of row i of rows."""
    if config.target == "probabilities" or config.options:
        return _probability_reference(config, rows, i)
    loss = config.loss
    row, index = _blocks(config, rows, i)
    ref = reference(loss, row)
    gradient = tuple(ref.gradient[b] for b in index)
    c = rows.classes
    allowance, value_allowance = [0.0] * c, 0.0
    spread = 1 + float(rows.logits[i].max() - rows.logits[i].min())
    if loss.name in HINGES:
        value_allowance = spread
        if loss.gamma > 0:
            allowance = [math.sqrt(c) * spread / loss.gamma] * c
    if loss.name in PROXIMAL:
        norm = float(mp.fsum(abs(g) for g in gradient))
        allowance = [norm] * c
    if loss.name in DISTILLATION and ref.scale:
        # (p + q) / T, of the represented student-teacher gap where it is below one.
        gap = float(np.abs(rows.logits[i] - rows.teacher[i]).max()) / loss.temperature
        allowance = [float(ref.scale[b]) * min(1.0, gap) for b in index]
    if config.target == "teacher":
        targets = (int(np.argmax(rows.teacher[i])),)
    elif config.multilabel:
        targets = tuple(int(j) for j in np.flatnonzero(rows.mask[i]))
    else:
        targets = (int(rows.labels[i]),)
    kinked = loss.name in HINGES and loss.gamma == 0 and ref.kink == 0
    return Exact(ref.value, gradient, tuple(allowance), value_allowance, targets, kinked)


# ---------------------------------------------------------------------------
# Backends.


@dataclass(frozen=True)
class Backend:
    name: str
    framework: str  # numpy, torch or jax
    device: str  # cpu or cuda
    option: str  # the backend= option ("kernel" or "composed"; NumPy: "kernel")

    @property
    def cuda(self) -> bool:
        return self.device == "cuda"


BACKENDS = (
    Backend("numpy", "numpy", "cpu", "kernel"),
    Backend("torch-cpu-kernel", "torch", "cpu", "kernel"),
    Backend("torch-cpu-composed", "torch", "cpu", "composed"),
    Backend("jax-cpu-kernel", "jax", "cpu", "kernel"),
    Backend("jax-cpu-composed", "jax", "cpu", "composed"),
    Backend("torch-cuda-kernel", "torch", "cuda", "kernel"),
    Backend("torch-cuda-composed", "torch", "cuda", "composed"),
    Backend("jax-cuda-kernel", "jax", "cuda", "kernel"),
    Backend("jax-cuda-composed", "jax", "cuda", "composed"),
)


def output_dtypes(backend: Backend, dtype: str) -> tuple[str, str, str]:
    """(work, value, gradient) dtypes of a call at the default precision: NumPy returns
    float64 losses computed in its work and gradients in its storage (float32 for 16-bit)."""
    wide = "float64" if dtype == "float64" else "float32"
    if backend.framework == "numpy":
        return wide, wide, wide
    return wide, wide, dtype


@functools.cache
def unavailable(backend: Backend) -> str | None:
    if backend.framework != "numpy" and importlib.util.find_spec(backend.framework) is None:
        return f"{backend.framework} is not installed"
    if backend.framework == "torch" and backend.cuda:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return "CUDA is unavailable"
    if backend.framework == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        if backend.cuda and not any(d.platform == "gpu" for d in jax.devices()):
            return "no JAX GPU"
    return None


def _numpy_call(config: Config, dtype: str, rows: Rows) -> tuple[np.ndarray, np.ndarray]:
    z = rows.logits.astype(_numpy_dtype(dtype))
    params = dict(
        k=config.loss.k,
        gamma=config.loss.gamma,
        alpha=config.loss.alpha,
        temperature=config.loss.temperature,
    )
    instance = named_losses.numpy_loss(config.name, **params)
    target: Any
    if config.target == "probabilities":
        target = libsdca.Probabilities(rows.probabilities)
    elif config.target == "teacher":
        target = rows.teacher.astype(_numpy_dtype(dtype))
    elif config.multilabel:
        target = rows.mask
    else:
        target = rows.labels
    values, gradient = libsdca.loss_and_grad(z, target, instance)
    return np.asarray(values, np.float64), np.asarray(gradient, np.float64)


def _options(config: Config) -> dict[str, Any]:
    loss = config.loss
    if config.name == "distillation":
        return dict(temperature=loss.temperature)
    if config.name == "proximal_distillation":
        return dict(temperature=loss.temperature, gamma=loss.alpha)
    return named_losses.parameters(config.name, loss.k, loss.gamma, loss.alpha)


def _function(module: Any, config: Config) -> Callable[..., Any]:
    if config.name in DISTILLATION:
        return getattr(module, config.name + "_loss")
    return named_losses.function(module, config.name)


def _torch_call(
    backend: Backend, config: Config, dtype: str, rows: Rows
) -> tuple[np.ndarray, np.ndarray]:
    L, torch = importlib.import_module("libsdca.torch"), importlib.import_module("torch")
    kind, device = getattr(torch, dtype), backend.device

    def tensor(x: np.ndarray) -> Any:
        return torch.tensor(x, device=device).to(kind)

    z = tensor(rows.logits).requires_grad_()
    options = dict(_options(config), reduction="none", backend=backend.option)
    if config.weighted:
        options["weight"] = tensor(np.array(WEIGHTS[: rows.classes]))
    if config.smoothing:
        options["label_smoothing"] = config.smoothing
    if config.target == "teacher":
        target = tensor(rows.teacher)
    elif config.target == "probabilities":
        target = tensor(rows.probabilities)
    elif config.multilabel:
        target = torch.tensor(rows.mask, device=device)
    else:
        target = torch.tensor(rows.labels, device=device)
    value = _function(L, config)(z, target, **options)
    value.backward(torch.ones_like(value))
    assert z.grad is not None
    return value.detach().double().cpu().numpy(), z.grad.double().cpu().numpy()


def _jax_call(
    backend: Backend, config: Config, dtype: str, rows: Rows
) -> tuple[np.ndarray, np.ndarray]:
    jax, jnp = importlib.import_module("jax"), importlib.import_module("jax.numpy")
    J = importlib.import_module("libsdca.jax")
    device = jax.devices("gpu" if backend.cuda else "cpu")[0]
    kind = jnp.dtype(dtype)

    def put(x: np.ndarray, as_dtype: Any = None) -> Any:
        return jax.device_put(np.asarray(x, dtype=as_dtype or kind), device)

    options = dict(_options(config), reduction="none", backend=backend.option)
    if config.weighted:
        options["weight"] = put(np.array(WEIGHTS[: rows.classes]))
    if config.smoothing:
        options["label_smoothing"] = config.smoothing
    if config.target == "teacher":
        target = put(rows.teacher)
    elif config.target == "probabilities":
        target = put(rows.probabilities)
    elif config.multilabel:
        target = put(rows.mask, bool)
    else:
        target = put(rows.labels, np.int64)
    fn = _function(J, config)
    with jax.default_device(device):
        loss = jax.jit(lambda z: fn(z, target, **options))
        value, pullback = jax.vjp(loss, put(rows.logits))
        (gradient,) = pullback(jnp.ones_like(value))
    return np.asarray(value, np.float64), np.asarray(gradient, np.float64)


def call(backend: Backend, config: Config, dtype: str, rows: Rows) -> tuple[np.ndarray, ...]:
    """Per-row losses and the logits' gradient, as float64."""
    if backend.framework == "numpy":
        return _numpy_call(config, dtype, rows)
    if backend.framework == "torch":
        return _torch_call(backend, config, dtype, rows)
    return _jax_call(backend, config, dtype, rows)


def skip_reason(backend: Backend, config: Config) -> str | None:
    if backend.framework == "numpy" and config.options:
        return "loss_and_grad takes no label_smoothing or class weights"
    return None


def params(configs: Sequence[Config]) -> list[Any]:
    out = []
    for backend in BACKENDS:
        for config in configs:
            for dtype in DTYPES:
                marks: list[Any] = [pytest.mark.cuda] if backend.cuda else []
                key = f"{config.label}-{backend.name}-{dtype}"
                out.append(pytest.param(backend, config, dtype, id=key, marks=marks))
    return out


# ---------------------------------------------------------------------------
# Checks.


@dataclass
class Report:
    """The failures of a test, one line each with its repro."""

    backend: Backend
    config: Config
    dtype: str
    lines: list[str]

    def fail(self, rows: Rows, i: int, what: str) -> None:
        z = ", ".join(f"{v:.9g}" for v in rows.logits[i])
        target = self.target_of(rows, i)
        self.lines.append(
            f"{self.config.label} {self.backend.name} {self.dtype} [{rows.tags[i]}] "
            f"logits=[{z}] {target}: {what}"
        )

    def target_of(self, rows: Rows, i: int) -> str:
        if self.config.target == "teacher":
            return "teacher=[" + ", ".join(f"{v:.9g}" for v in rows.teacher[i]) + "]"
        if self.config.target == "probabilities":
            return "p=[" + ", ".join(f"{v:.9g}" for v in rows.probabilities[i]) + "]"
        if self.config.multilabel:
            return f"positives={np.flatnonzero(rows.mask[i]).tolist()}"
        return f"label={int(rows.labels[i])}"

    def raise_if_failed(self, limit: int = 40) -> None:
        if self.lines:
            shown = self.lines[:limit]
            more = len(self.lines) - len(shown)
            tail = [f"... {more} more"] if more else []
            pytest.fail(f"{len(self.lines)} failures:\n" + "\n".join(shown + tail))


def _bound(x: MPReal, out: str, work: str, allowance: float, factor: int) -> MPReal:
    return factor * (eps(out) * abs(x) + eps(work) * mp.mpf(allowance)) + FORMATS[out][2]


def _ulps(actual: float, x: MPReal, out: str) -> str:
    """The error of actual in units of eps_out |x| (of eta_out below the normal range)."""
    _, tiny, eta, _ = FORMATS[out]
    unit = eps(out) * abs(x) if abs(x) >= tiny else mp.mpf(eta)
    return f"{mp.nstr(abs(mp.mpf(actual) - x) / unit, 3)} ulps"


def _flush(report: Report, out: str) -> float:
    """The magnitude below which a documented flush may return 0 in dtype out: the output
    dtype's normal range in JAX, the work's normal range in the proximal losses."""
    work = output_dtypes(report.backend, report.dtype)[0]
    jax = FORMATS[out][1] if report.backend.framework == "jax" else 0.0
    proximal = FORMATS[work][1] if report.config.name in PROXIMAL else 0.0
    return max(jax, proximal)


def _flushed(report: Report, actual: float, x: MPReal, out: str) -> bool:
    return actual == 0 and abs(x) < _flush(report, out)


def _vector(g: np.ndarray) -> str:
    return "[" + ", ".join(repr(float(x)) for x in g) + "]"


def check_row(
    report: Report,
    rows: Rows,
    i: int,
    ref: Exact,
    value: float,
    gradient: np.ndarray,
) -> None:
    work, value_dtype, out = output_dtypes(report.backend, report.dtype)
    if not (math.isfinite(value) and np.isfinite(gradient).all()):
        report.fail(rows, i, f"nonfinite loss {value!r} or gradient {_vector(gradient)}")
        return
    expected = f"exact {mp.nstr(ref.value, 17)}"
    if (
        not rounds_to_zero(ref.value, value_dtype)
        and value <= 0
        and not _flushed(report, value, ref.value, value_dtype)
    ):
        report.fail(rows, i, f"loss {value!r}, {expected}: the loss underflows")
    elif abs(mp.mpf(value) - ref.value) > _bound(
        ref.value, value_dtype, work, ref.value_allowance, BUDGET
    ) and not _flushed(report, value, ref.value, value_dtype):
        report.fail(rows, i, f"loss {value!r}, {expected} ({_ulps(value, ref.value, value_dtype)})")
    c = len(gradient)
    total = math.fsum(gradient.tolist())
    size = math.fsum(abs(g) for g in gradient.tolist())
    limit = c * eps(out) / 2 * size + c * (FORMATS[out][2] + _flush(report, out))
    if not ref.strict:
        limit += ULPS * eps(work) * math.fsum(ref.allowance)
    if abs(total) > limit:
        report.fail(
            rows,
            i,
            f"gradient {_vector(gradient)} sums to {total:.3g} = {total / size:.3g} |g|_1 "
            f"(bound {limit / size:.3g} |g|_1)",
        )
    if ref.kinked:
        return
    exact_gradient = "[" + ", ".join(str(mp.nstr(x, 17)) for x in ref.gradient) + "]"
    for j in ref.targets if ref.strict else ():
        x, actual = ref.gradient[j], float(gradient[j])
        if not rounds_to_zero(x, out) and actual == 0 and not _flushed(report, actual, x, out):
            others = np.delete(gradient, j)
            kind = "Softmax Collapse" if np.any(others != 0) else "the gradient underflows"
            report.fail(rows, i, f"gradient {_vector(gradient)}, exact {exact_gradient}: {kind}")
            return
    for j, (x, a) in enumerate(zip(ref.gradient, ref.allowance, strict=True)):
        actual = float(gradient[j])
        factor = ULPS if j in ref.targets else BUDGET
        if abs(mp.mpf(actual) - x) > _bound(x, out, work, a, factor) and not _flushed(
            report, actual, x, out
        ):
            report.fail(
                rows,
                i,
                f"gradient[{j}] {actual!r}, exact {mp.nstr(x, 17)} ({_ulps(actual, x, out)}); "
                f"gradient {_vector(gradient)}",
            )


def run_rows(report: Report, rows: Rows) -> None:
    values, gradients = call(report.backend, report.config, report.dtype, rows)
    for i in range(len(rows.logits)):
        check_row(report, rows, i, exact(report.config, rows, i), values[i], gradients[i])


@pytest.mark.parametrize(("backend", "config", "dtype"), params(CONFIGS))
def test_collapse(backend: Backend, config: Config, dtype: str) -> None:
    reason = unavailable(backend) or skip_reason(backend, config)
    if reason:
        pytest.skip(reason)
    report = Report(backend, config, dtype, [])
    for extreme in (False, True):
        for rows in collapse_rows(config, dtype, extreme):
            run_rows(report, rows)
    report.raise_if_failed()


@pytest.mark.parametrize(("backend", "config", "dtype"), params(CANCELLATION))
def test_cancellation(backend: Backend, config: Config, dtype: str) -> None:
    reason = unavailable(backend) or skip_reason(backend, config)
    if reason:
        pytest.skip(reason)
    report = Report(backend, config, dtype, [])
    rows, gaps = cancellation_rows(dtype)
    values, gradients = call(backend, config, dtype, rows)
    for i, gap in enumerate(gaps):
        if gap == 0 and config.target == "teacher" and np.any(gradients[i] != 0):
            largest = np.abs(gradients[i]).max()
            report.fail(rows, i, f"gradient not exactly 0 at g = 0 (max |g| {largest:.3g})")
            continue
        check_row(report, rows, i, exact(config, rows, i), values[i], gradients[i])
    report.raise_if_failed()


@pytest.mark.parametrize("framework", ["jax", "torch"])
def test_distillation_curvature_at_equal_logits(framework: str) -> None:
    """At q = p the first derivative vanishes and the Hessian is the softmax covariance."""
    t = np.array([[0.5, 1.0, -0.5], [0.0, -np.inf, 2.0]], dtype=np.float64)
    direction = np.array([[1.0, -2.0, 0.5], [-0.5, 1.0, 1.0]])
    temperature = 2.0
    options = dict(temperature=temperature, reduction="sum", backend="composed")
    if framework == "jax":
        jax = pytest.importorskip("jax")
        jnp = importlib.import_module("jax.numpy")
        loss = importlib.import_module("libsdca.jax").distillation_loss
        with jax.enable_x64(True):
            teacher = jnp.asarray(t)
            gradient, product = jax.jvp(
                jax.grad(lambda z: loss(z, teacher, **options)),
                (teacher,),
                (jnp.asarray(direction),),
            )
    else:
        torch = pytest.importorskip("torch")
        loss = importlib.import_module("libsdca.torch").distillation_loss
        z = torch.tensor(t, requires_grad=True)
        (gradient,) = torch.autograd.grad(loss(z, z.detach(), **options), z, create_graph=True)
        (product,) = torch.autograd.grad(gradient, z, torch.tensor(direction))
        gradient = gradient.detach().numpy()
    prob = np.exp(t / temperature)
    prob /= prob.sum(-1, keepdims=True)
    expected = prob * (direction - (prob * direction).sum(-1, keepdims=True)) / temperature**2
    np.testing.assert_array_equal(gradient, 0.0)
    np.testing.assert_allclose(product, expected, atol=1e-15)

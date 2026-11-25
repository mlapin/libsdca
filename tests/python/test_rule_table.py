"""Every backend against the rule table (rule_table.py): the C ABI, NumPy, Rust, the C++ kernels
(sdca::kernel::evaluate) and sdca/loss.h, the kernels of PyTorch and JAX, their composed
backends, and the CUDA kernels of both under the cuda marker.

The rows run in batches, one call per backend for the rows of one loss, parameters, options
and dtype. A row's loss and gradient match the table: its exact entries (zeros with their
sign, infinities, NaN, where the rule fixes them: Row.exact) by bytes, the others within
TOLERANCE of the references (COMPOSED_TOLERANCE for the composed backends, whose JAX calls
are jitted); a row with same_as gives the bytes of that row. The reports of invalid
rows follow docs/API.md (Errors): the C ABI's code, count, first row and message, one
InvalidRowsWarning per NumPy call with the C ABI's message, Rust's report, the C++ kernel's
result, sdca/loss.h's exception, the frameworks' silence; check_rows refuses the call before
writing, with the same message. Each call of ERRORS gives the text of rule_table.CANONICAL in
every interface that can make it.

Where a backend disagrees with the table, FINDINGS names it (the cases xfail, not strict) with
a strict minimal repro below. Interfaces leave out what they do not take by design (why_not).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import interfaces
import libsdca
import masked_cases as M
import numpy as np
import pytest
from interfaces import NATIVE_DTYPES, PROBABILITIES, Session
from libsdca import _frontend as f
from rule_table import ACTIVE_ERRORS, IGNORE, ROWS, Error, Row, canonical
from storage_limits import LIMITS

BACKENDS = (
    "capi",
    "numpy",
    "rust",
    "cpp_kernel",
    "cpp_loss",
    "torch",
    "jax",
    "torch_composed",
    "jax_composed",
    "torch_cuda",
    "jax_cuda",
)
NATIVE = ("capi", "numpy", "rust", "cpp_loss")
COMPILED = ("rust", "cpp_kernel", "cpp_loss")
# The tolerances of the reference entries, relative to 1 + |expected|, in eps of the work type
# (float64 for float64 storage, else float32) and roundoffs of a 16-bit storage:
KERNEL_EPS = 16  # a row's few roundings in the work (the kernels are within 2.6 eps)
COMPOSED_EPS = 64  # unfused tensor operations round every step (within 6.8 eps)
STORAGE_ROUNDOFFS = 1  # a 16-bit storage: the gradient rounded once to it


def _tolerance(dtype: str, work_eps: int) -> float:
    work = LIMITS["float64" if dtype == "float64" else "float32"]
    storage = LIMITS[dtype].roundoff * STORAGE_ROUNDOFFS if dtype in ("float16", "bfloat16") else 0
    return work_eps * work.eps + storage


TOLERANCE = {dtype: _tolerance(dtype, KERNEL_EPS) for dtype in LIMITS}
COMPOSED_TOLERANCE = {dtype: _tolerance(dtype, COMPOSED_EPS) for dtype in LIMITS}


# --- batches ---------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Batch:
    """The rows of one call: one loss, parameters, options, dtype and width."""

    id: str
    rows: tuple[Row, ...]

    @property
    def first(self) -> Row:
        return self.rows[0]

    @property
    def params(self) -> dict[str, float]:
        return self.first.options

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def invalid(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if not row.valid]

    @property
    def target_kind(self) -> str:
        loss = self.first.loss
        if loss in interfaces.DISTILLATION:
            return "teacher"
        if loss == PROBABILITIES:
            return "probs"
        return "mask" if loss in M.MULTILABEL else "labels"

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The scores (n, classes) float64 and the target of the rows."""
        z = np.array([row.logits for row in self.rows], np.float64)
        kind = self.target_kind
        if kind == "labels":
            return z, np.array([row.target for row in self.rows], np.int64)
        if kind == "mask":
            mask = np.zeros(z.shape, bool)
            for i, row in enumerate(self.rows):
                mask[i, list(row.target)] = True
            return z, mask
        return z, np.array([row.target for row in self.rows], np.float64)


def _batches() -> list[Batch]:
    groups: dict[tuple[Any, ...], list[Row]] = {}
    for row in ROWS:
        key = (row.loss, row.params, row.dtype, row.classes, row.weight, row.ignore_index)
        key += (row.label_smoothing, row.upstream, bool(row.native_error))
        groups.setdefault(key, []).append(row)
    return [
        Batch(f"b{i:03d}-{rows[0].loss}", tuple(rows)) for i, rows in enumerate(groups.values())
    ]


BATCHES = _batches()


def why_not(batch: Batch, backend: str) -> str | None:
    """Why a backend does not take a batch, by design, or None."""
    row = batch.first
    if backend in NATIVE and row.upstream != 1.0:
        return "upstream factors are the frameworks' (their backward pass)"
    if backend == "cpp_loss" and row.has_options:
        return "sdca/loss.h takes no class weights, ignore_index or label smoothing"
    if backend in ("capi", "rust", "cpp_loss") and row.dtype not in NATIVE_DTYPES:
        return "the C ABI, Rust and sdca/loss.h store float32 and float64 only"
    if backend == "cpp_loss" and row.loss == PROBABILITIES:
        return "sdca/loss.h takes no target probabilities"
    if (
        backend == "rust"
        and batch.target_kind == "labels"
        and min(r.target for r in batch.rows) < 0
    ):
        return "Rust labels are usize"
    if backend == "jax_composed" and row.name.endswith("-f32"):
        return "time budget: each JAX call compiles its programs (about 0.2 s); float64 only"
    weight = row.weight or ()
    if backend == "cpp_kernel" and not all(math.isfinite(w) and w >= 0 for w in weight):
        return "the NaN rule of class weights is the frontends' (_frontend), not sdca/kernel.h's"
    return None


def _cuda(backend: str) -> list[Any]:
    if not backend.endswith("_cuda"):
        return []
    return [pytest.mark.cuda]


def _params(cases: list[Any], backends: tuple[str, ...]) -> list[Any]:
    return [
        pytest.param(case, backend, id=f"{case.id}-{backend}", marks=_cuda(backend))
        for case in cases
        for backend in backends
        if why_not(case, backend) is None
    ]


# --- the compiled interfaces' inputs ---------------------------------------------------------


def _prepare(directory: Path) -> None:
    lines = []
    for batch in BATCHES:
        row = batch.first
        z, target = batch.arrays()
        kind = batch.target_kind
        interfaces.write_inputs(
            directory,
            batch.id,
            row.dtype,
            z,
            labels=target if kind == "labels" else None,
            mask=target if kind == "mask" else None,
            probs=target if kind == "probs" else None,
            teacher=target if kind == "teacher" else None,
            weight=None if row.weight is None else np.array(row.weight),
        )
        common = dict(
            **interfaces.c_parameters(row.loss, batch.params),
            label_smoothing=row.label_smoothing,
            ignore_index=row.ignore_index,
            weight=row.weight is not None,
            threads=1,
            upstream=row.upstream,
        )
        common["k"] = common["k"] or 1
        kind_id = interfaces.C_KINDS[row.loss]
        args = (kind_id, row.dtype, batch.n, row.classes, kind)
        lines.append(interfaces.manifest_line(batch.id, batch.id, *args, **common))
        if batch.invalid:
            check = f"{batch.id}-check"
            lines.append(
                interfaces.manifest_line(check, batch.id, *args, **common, check_rows=True)
            )
    rust_lines = []
    for error in ACTIVE_ERRORS:
        if not _rust_call(error) and not _c_call(error):
            continue
        n, classes = error.shape
        key = f"error-{error.name}"
        z = np.zeros(error.shape)
        loss, target = error.loss, _error_target(error, n, classes)
        kind = "teacher" if loss in interfaces.DISTILLATION else "labels"
        kind = "mask" if loss in M.MULTILABEL else kind
        kind = "probs" if loss == PROBABILITIES else kind
        options = _error_options(error)
        interfaces.write_inputs(
            directory, key, "float64", z,
            labels=target if kind == "labels" else None,
            mask=target if kind == "mask" else None,
            probs=target if kind == "probs" else None,
            teacher=target if kind == "teacher" else None,
            weight=options.get("weight"),
        )  # fmt: skip
        common = interfaces.c_parameters(loss, dict(error.params))
        args = (interfaces.C_KINDS[loss], "float64", n, classes, kind)
        line = interfaces.manifest_line(
            key, key, *args, **common, threads=1,
            label_smoothing=options.get("label_smoothing", 0.0),
            ignore_index=options.get("ignore_index"),
            weight="weight" in options,
        )  # fmt: skip
        # The C++ helper reads the sizes the case implies: the size errors go to Rust alone.
        (lines if _c_call(error) else rust_lines).append(line)
    (directory / "cases.txt").write_text("".join(line + "\n" for line in lines))
    (directory / "rust_cases.txt").write_text("".join(line + "\n" for line in rust_lines))


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    return interfaces.shared_session(tmp_path_factory, "rule_table", _prepare)


# --- outputs of the backends ------------------------------------------------------------------


@dataclasses.dataclass
class Outputs:
    values: np.ndarray  # float64 of the backend's value dtype (n,)
    grads: np.ndarray  # float64 of the storage (n, classes); NaN where not written
    value_dtype: str  # float64, or float32 (the frameworks' values of non-float64 logits)
    report: tuple[int, int] | None = None  # (invalid rows, first) where the backend reports it
    message: str = ""  # the invalid-row message, where the backend gives one
    warnings: int = 0  # InvalidRowsWarnings of the call


def _framework(backend: str) -> tuple[str, str, str]:
    """(framework, backend option, device) of a framework backend."""
    framework = backend.split("_")[0]
    option = "composed" if backend.endswith("_composed") else "kernel"
    device = "cuda" if backend.endswith("_cuda") else "cpu"
    return framework, option, device


def _framework_options(row: Row, convert: Callable[[np.ndarray], Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if row.weight is not None:
        options["weight"] = convert(np.array(row.weight))
    if row.ignore_index is not None:
        options["ignore_index"] = row.ignore_index
    if row.label_smoothing:
        options["label_smoothing"] = row.label_smoothing
    return options


def run_torch(batch: Batch, backend: str) -> Outputs:
    torch = importlib.import_module("torch")
    L = importlib.import_module("libsdca.torch")
    _, option, device = _framework(backend)
    row = batch.first
    kind = getattr(torch, row.dtype)
    z, target = batch.arrays()
    leaf = torch.tensor(z, device=device).to(kind).requires_grad_()
    t = torch.tensor(target, device=device)
    t = t.to(kind) if t.is_floating_point() else t
    options = _framework_options(row, lambda a: torch.tensor(a, device=device).to(kind))
    if row.ignore_index is None and batch.target_kind == "labels" and backend.startswith("torch"):
        options["ignore_index"] = IGNORE  # PyTorch's default, said explicitly
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = interfaces.framework_call(
            L, row.loss, leaf, t, batch.params, reduction="none", backend=option, **options
        )
        value.backward(torch.full_like(value, row.upstream))
    assert leaf.grad is not None
    grads = leaf.grad.detach().cpu().double().numpy()
    count = sum(issubclass(w.category, libsdca.InvalidRowsWarning) for w in caught)
    dtype = "float64" if row.dtype == "float64" else "float32"
    return Outputs(value.detach().cpu().double().numpy(), grads, dtype, warnings=count)


def run_jax(batch: Batch, backend: str) -> Outputs:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    J = importlib.import_module("libsdca.jax")
    _, option, device = _framework(backend)
    row = batch.first
    x64 = row.dtype == "float64"
    z, target = batch.arrays()
    with (
        interfaces.x64(jax, x64),
        jax.default_device(jax.devices("gpu" if device == "cuda" else "cpu")[0]),
    ):
        kind = jnp.dtype(row.dtype)

        def floats(a: np.ndarray) -> Any:
            return jnp.asarray(a.astype(np.float64 if x64 else np.float32)).astype(kind)

        if target.dtype == np.float64:
            t = floats(target)
        else:
            t = jnp.asarray(target if target.dtype == bool or x64 else target.astype(np.int32))
        options = _framework_options(row, floats)

        def both(scores: Any) -> tuple[Any, Any]:
            value, pullback = jax.vjp(
                lambda s: interfaces.framework_call(
                    J, row.loss, s, t, batch.params, reduction="none", backend=option, **options
                ),
                scores,
            )
            return value, pullback(jnp.full_like(value, row.upstream))[0]

        # The kernels eagerly; the composed operations in one program (eagerly, each would
        # compile on its own).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value, grad = (jax.jit(both) if option == "composed" else both)(floats(z))
        values = np.asarray(value.astype(jnp.float32) if row.dtype != "float64" else value)
        grads = np.asarray(grad.astype(jnp.float32) if x64 is False else grad)
    count = sum(issubclass(w.category, libsdca.InvalidRowsWarning) for w in caught)
    dtype = "float64" if x64 else "float32"
    return Outputs(values.astype(np.float64), grads.astype(np.float64), dtype, warnings=count)


def run_capi(batch: Batch, check_rows: bool = False, fill: float = math.nan) -> interfaces.CResult:
    row = batch.first
    z, target = batch.arrays()
    return interfaces.c_loss_evaluate(
        interfaces.C_KINDS[row.loss],
        row.dtype,
        z,
        batch.target_kind,
        target,
        **interfaces.c_parameters(row.loss, batch.params),
        check_rows=check_rows,
        fill=fill,
        **row.options_of(),
    )


def _capi_outputs(batch: Batch) -> Outputs:
    result = run_capi(batch)
    assert result.code in (interfaces.SDCA_OK, interfaces.SDCA_INVALID_ROWS), result.message
    grads = np.asarray(result.grads, np.float64)
    report = (result.invalid_rows, result.first_invalid_row)
    return Outputs(result.values, grads, "float64", report, result.message)


def _numpy_target(batch: Batch, target: np.ndarray) -> Any:
    return libsdca.Probabilities(target) if batch.target_kind == "probs" else target


def run_numpy(batch: Batch, check_rows: bool = False) -> tuple[Outputs | None, str]:
    """NumPy's outputs, or None and the message of its error."""
    row = batch.first
    z, target = batch.arrays()
    stored = row.dtype if row.dtype in NATIVE_DTYPES else "float32"
    if batch.target_kind == "teacher":
        target = target.astype(stored)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            values, grads = libsdca.loss_and_grad(
                z.astype(stored),
                _numpy_target(batch, target),
                interfaces.numpy_loss(row.loss, batch.params),
                check_rows=check_rows,
                **row.options_of(),
            )
        except (ValueError, TypeError) as error:
            return None, str(error)
    found = [w for w in caught if issubclass(w.category, libsdca.InvalidRowsWarning)]
    # 16-bit storage: NumPy's float32 gradient rounded once to the 16-bit type.
    rounded = interfaces.values_of(interfaces.storage(grads, row.dtype).tobytes(), row.dtype)
    message = str(found[0].message) if found else ""
    out = Outputs(values, rounded.reshape(grads.shape), "float64", None, message, len(found))
    return out, ""


def _read_compiled(session: Session, batch: Batch, backend: str, id: str = "") -> Outputs | str:
    """The outputs of a compiled backend, or the message of its error."""
    id = id or batch.id
    session.require(backend)
    error = session.path(id, backend, "error")
    if error.exists():
        return error.read_text()
    dtype = batch.first.dtype
    values = np.frombuffer(session.path(id, backend, "values").read_bytes())
    grads = interfaces.values_of(session.path(id, backend, "grads").read_bytes(), dtype)
    grads = grads.reshape(batch.n, batch.first.classes)
    report = None
    if backend == "cpp_kernel":
        status, count, first = map(int, session.path(id, backend, "status").read_text().split())
        if status != 0:
            return f"status {status}"
        report = (count, first)
    elif backend == "rust":
        count, first = map(int, session.path(id, backend, "report").read_text().split())
        report = (count, first)
    return Outputs(values.copy(), grads, "float64", report)


def outputs(session: Session, batch: Batch, backend: str) -> Outputs | str:
    """A backend's outputs of a batch, or the message of the error it raised."""
    if backend in COMPILED:
        return _read_compiled(session, batch, backend)
    if backend == "capi":
        if batch.first.native_error:
            return run_capi(batch).message
        return _capi_outputs(batch)
    if backend == "numpy":
        found, message = run_numpy(batch)
        return found if found is not None else message
    framework = _framework(backend)[0]
    if importlib.util.find_spec(framework) is None:
        pytest.skip(f"{framework} is not installed")
    if backend.endswith("_cuda"):
        _require_cuda(framework)
    return run_torch(batch, backend) if framework == "torch" else run_jax(batch, backend)


def _require_cuda(framework: str) -> None:
    if framework == "torch":
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
    else:
        jax = importlib.import_module("jax")
        if not any(d.platform == "gpu" for d in jax.devices()):
            pytest.skip("no JAX GPU")


# --- comparisons ------------------------------------------------------------------------------


def entry_mismatches(
    actual: np.ndarray, want: np.ndarray, tol: float, fixed: np.ndarray | None = None
) -> np.ndarray:
    """Where actual is not want: NaN where want is NaN; the bytes (with the sign) where want
    is a zero or an infinity that the rule fixes (fixed; None: every one); else within tol
    relative to 1 + |want|."""
    nan = np.isnan(want)
    exact = (want == 0) | np.isinf(want)
    if fixed is not None:
        exact &= fixed
    same = (actual == want) & (np.signbit(actual) == np.signbit(want))
    with np.errstate(invalid="ignore"):
        near = np.abs(actual - want) <= tol * (1 + np.abs(want))
    ok = np.where(nan, np.isnan(actual), np.where(exact, same, near))
    return ~ok


def _rounded(x: np.ndarray, dtype: str) -> np.ndarray:
    return x.astype(np.float32).astype(np.float64) if dtype == "float32" else x


def compare_rows(batch: Batch, out: Outputs, backend: str) -> list[str]:
    """The rows whose loss or gradient is not the table's (or not their same_as row's)."""
    dtype = batch.first.dtype
    tol = (COMPOSED_TOLERANCE if "composed" in backend else TOLERANCE)[dtype]
    names = {row.name: i for i, row in enumerate(batch.rows)}
    errors = []
    for i, row in enumerate(batch.rows):
        want = _rounded(np.array([row.value]), out.value_dtype)
        if entry_mismatches(out.values[i : i + 1], want, tol).any():
            errors.append(f"{row.name} ({row.rule}): loss {out.values[i]!r}, not {want[0]!r}")
        if row.gradient is not None:
            g, w = out.grads[i], np.array(row.gradient) * 1.0
            fixed = None if row.exact is None else np.array(row.exact)
            bad = np.flatnonzero(entry_mismatches(g, w, tol, fixed))
            if len(bad):
                entries = ", ".join(f"{j}: {g[j]!r} not {w[j]!r}" for j in bad[:4])
                errors.append(f"{row.name} ({row.rule}): gradient {entries}")
        if row.same_as:
            j = names[row.same_as]
            same = out.values[i].tobytes() == out.values[j].tobytes()
            if not (same and out.grads[i].tobytes() == out.grads[j].tobytes()):
                errors.append(f"{row.name} ({row.rule}): not the bytes of {row.same_as}")
    return errors


def compare_report(batch: Batch, out: Outputs, backend: str) -> list[str]:
    """The invalid-row report of a backend against the batch's invalid rows."""
    invalid = batch.invalid
    count, first = len(invalid), invalid[0] if invalid else None
    errors = []
    if backend in ("capi", "cpp_kernel", "rust") and out.report is not None:
        none = {"capi": batch.n, "cpp_kernel": batch.n, "rust": -1}[backend]
        want = (count, first if first is not None else none)
        if out.report != want:
            errors.append(f"report {out.report}, not {want}")
    if backend == "capi" and count:
        prefix = f"{count} of {batch.n} rows are invalid (first: row {first}): "
        if not out.message.startswith(prefix):
            errors.append(f"message {out.message!r} does not start {prefix!r}")
    if backend == "numpy" and out.warnings != (1 if count else 0):
        errors.append(f"{out.warnings} InvalidRowsWarnings, not {1 if count else 0}")
    if backend == "numpy" and count:
        want_message = run_capi(batch).message
        if out.message != want_message:
            errors.append(f"warning {out.message!r}, not the C ABI's {want_message!r}")
    if backend.startswith(("torch", "jax")) and out.warnings:
        errors.append(f"{out.warnings} InvalidRowsWarnings: the frameworks' NaN rows are silent")
    return errors


# --- findings ---------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Finding:
    description: str
    subjects: tuple[str, ...]  # rows (by name) or ERRORS (by name) where it shows
    backends: tuple[str, ...]
    repro: str


PYTHON = ("numpy", "torch", "jax", "torch_composed", "jax_composed")
FRAMEWORKS = ("torch", "jax", "torch_composed", "jax_composed")
FINDINGS = {
    "jax_composed_flushed_negative_zero": Finding(
        "backend='composed' in JAX gives +0.0 for the label's entry of bfloat16 [0, -100, -100], "
        "-7.4e-44 in float32 work: XLA flushes that subnormal (docs/API.md, Known limits), so "
        "its sign is lost before the bfloat16 storage, where the kernels keep -0.0.",
        ("storage-zero-bfloat16",),
        ("jax_composed",),
        "test_finding_jax_composed_flushed_negative_zero",
    ),
}


def _verdict(subjects: tuple[str, ...], label: str, backend: str, errors: list[str]) -> None:
    message = f"{backend} on {label}:\n" + "\n".join(errors)
    found = [
        f"{name} ({f.repro})"
        for name, f in FINDINGS.items()
        if backend in f.backends and {s.removesuffix("-f32") for s in subjects} & set(f.subjects)
    ]
    if errors and found:
        pytest.xfail(f"{', '.join(found)}: {message}")
    assert not errors, message


def _names(batch: Batch) -> tuple[str, ...]:
    return tuple(row.name for row in batch.rows)


# --- tests ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("batch", "backend"), _params(BATCHES, BACKENDS))
def test_rows(session: Session, batch: Batch, backend: str) -> None:
    out = outputs(session, batch, backend)
    native_error = batch.first.native_error
    if native_error and backend in ("capi", "numpy", "rust"):
        got = out if isinstance(out, str) else "no error"
        errors = [] if got == native_error else [f"error {got!r}, not {native_error!r}"]
        _verdict(_names(batch), batch.id, backend, errors)
        return
    if isinstance(out, str):
        # sdca/loss.h throws for an invalid row (one row per call); nothing else may fail.
        expected = backend == "cpp_loss" and (batch.invalid or native_error)
        _verdict(_names(batch), batch.id, backend, [] if expected else [f"error: {out}"])
        return
    if backend == "cpp_loss" and batch.invalid:
        _verdict(_names(batch), batch.id, backend, ["no exception for an invalid row"])
    errors = compare_rows(batch, out, backend) + compare_report(batch, out, backend)
    _verdict(_names(batch), batch.id, backend, errors)


CHECKED = [b for b in BATCHES if b.invalid and not b.first.native_error]


@pytest.mark.parametrize(
    ("batch", "backend"), _params(CHECKED, ("capi", "numpy", "rust", "cpp_kernel"))
)
def test_check_rows(session: Session, batch: Batch, backend: str) -> None:
    """check_rows refuses a call with invalid rows before writing anything, with the message
    of the C ABI's invalid rows."""
    message = run_capi(batch).message
    errors = []
    if backend == "capi":
        result = run_capi(batch, check_rows=True, fill=7.0)
        if result.code != interfaces.SDCA_INVALID_ARGUMENT:
            errors.append(f"code {result.code}, not SDCA_INVALID_ARGUMENT")
        if not ((result.values == 7.0).all() and (result.grads == 7.0).all()):
            errors.append("outputs written")
        if result.message != message:
            errors.append(f"message {result.message!r}, not {message!r}")
    elif backend == "numpy":
        found, got = run_numpy(batch, check_rows=True)
        if found is not None or got != message:
            errors.append(f"error {got!r}, not ValueError {message!r}")
    elif backend == "rust":
        got = _read_compiled(session, batch, "rust", f"{batch.id}-check")
        if got != message:
            errors.append(f"error {got if isinstance(got, str) else 'none'!r}, not {message!r}")
    else:
        session.require("cpp_kernel")
        status = session.path(f"{batch.id}-check", "cpp_kernel", "status").read_text().split()
        values = session.path(f"{batch.id}-check", "cpp_kernel", "values").read_bytes()
        grads = session.path(f"{batch.id}-check", "cpp_kernel", "grads").read_bytes()
        if status[0] != "-2":
            errors.append(f"status {status[0]}, not status_invalid_argument")
        if set(values) | set(grads) != {interfaces.UNWRITTEN}:
            errors.append("outputs written")
    _verdict(_names(batch), f"{batch.id}-check", backend, errors)


# --- errors -----------------------------------------------------------------------------------

# sdca/kernel.h (cpp_kernel) leaves the checks of parameters to its callers.
ERROR_BACKENDS = (
    "capi",
    "numpy",
    "rust",
    "cpp_loss",
    "torch",
    "jax",
    "torch_composed",
    "jax_composed",
)


LABEL_TARGETS = ("label_above", "label_negative")
# The size errors Rust takes: its labels and target matrices carry their sizes.
RUST_TARGETS = ("labels_length", "teacher_shape", "probs_shape", "label_above")


def _error_options(error: Error) -> dict[str, Any]:
    """The weight (an array), label_smoothing and ignore_index of an error's call."""
    options = dict(error.options)
    if "weight" in options:
        options["weight"] = np.array(options["weight"], np.float64)
    return options


def _c_call(error: Error) -> bool:
    """Whether the C ABI makes the call: parameters, widths, label values and options, no
    target sizes or weights of another length (it takes pointers)."""
    weight = dict(error.options).get("weight")
    fits = weight is None or len(weight) == error.shape[1]
    return error.target in ("", *LABEL_TARGETS) and fits


def _rust_call(error: Error) -> bool:
    ignored = dict(error.options).get("ignore_index", 0)
    return error.target in ("", *RUST_TARGETS) and ignored >= 0


def error_why_not(error: Error, backend: str) -> str | None:
    if backend in ("capi", "cpp_loss") and not _c_call(error):
        return "the C ABI and sdca/loss.h take pointers: no target sizes"
    if backend == "cpp_loss" and error.options:
        return "sdca/loss.h takes no class weights, ignore_index or label smoothing"
    if backend == "rust" and not _rust_call(error):
        return "Rust takes no masks, float labels or labels below 0"
    if backend == "cpp_loss" and error.loss == PROBABILITIES:
        return "sdca/loss.h takes no target probabilities"
    if backend in FRAMEWORKS and error.name in VALUE_ERRORS:
        return "PyTorch and JAX check no values of arrays (traced in JAX): NaN rows"
    return None


def _error_target(error: Error, n: int, classes: int) -> np.ndarray:
    loss, kind = error.loss, error.target
    if kind == "labels_length":
        return np.zeros(n + 1, np.int64)
    if kind == "float_labels":
        return np.zeros(n, np.float64)
    if kind == "label_above":
        return np.full(n, classes, np.int64)
    if kind == "label_negative":
        return np.full(n, -1, np.int64)
    if loss in interfaces.DISTILLATION:
        return np.zeros((n, classes - 1 if kind == "teacher_shape" else classes))
    if loss in M.MULTILABEL:
        mask = np.zeros((n, classes - 1 if kind == "mask_shape" else classes), bool)
        mask[:, 0] = True
        return mask
    if loss == PROBABILITIES:
        return np.full((n, classes - 1 if kind == "probs_shape" else classes), 1 / classes)
    return np.zeros(n, np.int64)


def error_message(session: Session, error: Error, backend: str) -> str:
    """The message of the backend's error ("" if it raised none; the type for Python)."""
    n, classes = error.shape
    params = dict(error.params)
    z = np.zeros((n, classes))
    target = _error_target(error, n, classes)
    if backend == "capi":
        kind = "teacher" if error.loss in interfaces.DISTILLATION else "labels"
        kind = "mask" if error.loss in M.MULTILABEL else kind
        kind = "probs" if error.loss == PROBABILITIES else kind
        result = interfaces.c_loss_evaluate(
            interfaces.C_KINDS[error.loss], "float64", z, kind,
            z if kind == "teacher" else target, **interfaces.c_parameters(error.loss, params),
            **_error_options(error),
        )  # fmt: skip
        return result.message if result.code == interfaces.SDCA_INVALID_ARGUMENT else ""
    if backend in COMPILED:
        session.require(backend)
        key = f"error-{error.name}"
        path = session.path(key, backend, "error")
        return path.read_text() if path.exists() else ""
    if backend == "numpy":
        try:
            loss = interfaces.numpy_loss(error.loss, params)
            t = libsdca.Probabilities(target) if error.loss == PROBABILITIES else target
            libsdca.loss_and_grad(z, t, loss, **_error_options(error))
        except (ValueError, TypeError, IndexError) as found:
            return f"{type(found).__name__}: {found}"
        return ""
    framework, option, _ = _framework(backend)
    if importlib.util.find_spec(framework) is None:
        pytest.skip(f"{framework} is not installed")
    module = importlib.import_module(f"libsdca.{framework}")
    options = dict(error.options)
    if framework == "torch":
        torch = importlib.import_module("torch")
        convert: Callable[[np.ndarray], Any] = torch.tensor
    else:
        jnp = importlib.import_module("jax.numpy")
        convert = jnp.asarray
    if "weight" in options:
        options["weight"] = convert(np.array(options["weight"]))
    try:
        interfaces.framework_call(
            module, error.loss, convert(z), convert(target), params, backend=option, **options
        )
    except (ValueError, TypeError, IndexError) as found:
        return f"{type(found).__name__}: {found}"
    return ""


TYPE_ERRORS = ("float-labels", "smoothing-hinge", "weight-multilabel", "ignore-distillation")
# The errors of values of arrays, which the frameworks do not check.
VALUE_ERRORS = ("weight-nan",)


@pytest.mark.parametrize(
    ("error", "backend"),
    [
        pytest.param(e, b, id=f"{e.name}-{b}")
        for e in ACTIVE_ERRORS
        for b in ERROR_BACKENDS
        if error_why_not(e, b) is None
    ],
)
def test_errors(session: Session, error: Error, backend: str) -> None:
    """Every interface rejects the call with the canonical text and exception type."""
    got = error_message(session, error, backend)
    dtype = "float64"
    if backend.startswith("jax"):
        # float64 labels are float32 to JAX unless x64 is on (other tests may leave it on).
        dtype = str(importlib.import_module("jax.numpy").asarray(np.zeros(1)).dtype)
    want = canonical(error, backend, dtype)
    if backend in ("numpy", *FRAMEWORKS):
        kind = "TypeError" if error.name in TYPE_ERRORS else "ValueError"
        if backend in FRAMEWORKS and error.name in ("label-above", "label-negative"):
            kind = "IndexError"
        want = f"{kind}: {want}"
    assert got == want, f"{backend} on error-{error.name}: {got!r}, not {want!r}"


def test_table_is_consistent() -> None:
    """Every row of a batch has the batch's loss and width, and every invalid row NaN
    gradients; rows the table cannot run anywhere are named by why_not."""
    for batch in BATCHES:
        for row in batch.rows:
            assert row.classes == batch.first.classes
            if not row.valid:
                assert row.gradient is not None and all(math.isnan(g) for g in row.gradient)
        assert any(why_not(batch, b) is None for b in BACKENDS)


# --- minimal repros of FINDINGS: strict xfail, they pass once fixed --------------------------


def _strict(name: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=FINDINGS[name].description)


def _torch_composed(z: np.ndarray, dtype: str, label: int, **options: Any) -> Any:
    torch = pytest.importorskip("torch")
    L = importlib.import_module("libsdca.torch")
    leaf = torch.tensor(z).to(getattr(torch, dtype)).requires_grad_()
    options = {"reduction": "none", "backend": "composed", **options}
    value = L.cross_entropy(leaf, torch.tensor([label]), **options)
    value.backward(torch.ones_like(value))
    return value.detach().double().numpy(), leaf.grad.double().numpy()


def test_composed_16bit_negative_zero() -> None:
    # docs/API.md (Signed zeros): a negative entry that rounds to zero only in float16 storage
    # stays -0.0, as the kernels give it.
    _, grad = _torch_composed(np.array([[0.0, -20.0, -20.0]]), "float16", 0)
    assert np.signbit(grad[0, 0]), grad


@_strict("jax_composed_flushed_negative_zero")
def test_finding_jax_composed_flushed_negative_zero() -> None:
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    J = importlib.import_module("libsdca.jax")
    z = jnp.asarray([[0.0, -100.0, -100.0]], jnp.bfloat16)
    grad = jax.grad(lambda x: J.cross_entropy(x, jnp.asarray([0]), backend="composed"))(z)
    assert np.signbit(np.asarray(grad, np.float32)[0, 0]), grad  # +0.0 now


def test_composed_zero_weight_masked_label() -> None:
    # docs/API.md (Masked classes): a masked label of class weight 0 is an invalid row.
    torch = pytest.importorskip("torch")
    weight = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    value, grad = _torch_composed(np.array([[-np.inf, 1.0, 0.5]]), "float64", 0, weight=weight)
    assert np.isnan(value).all() and np.isnan(grad).all(), (value, grad)


# --- the options table -------------------------------------------------------------------------

_OPTION_VALUES = {"weight": np.ones(4), "label_smoothing": 0.25, "ignore_index": 1}


@pytest.mark.parametrize("option", sorted(_OPTION_VALUES))
@pytest.mark.parametrize("name", sorted(f.PARAMETERS))
def test_options_table(name: str, option: str) -> None:
    """The losses that take each option are those of _frontend.OPTIONS in every interface: the
    C ABI (its kinds), NumPy, and the signatures of the PyTorch and JAX functions."""
    takes = name in f.OPTIONS[option]
    kinds = [f"topk_hinge_{v}" for v in f.VARIANTS] if name == "topk_hinge" else [name]
    z, value = np.zeros((2, 4)), _OPTION_VALUES[option]
    target_kind = "teacher" if name in f.DISTILLATION else "labels"
    target_kind = "mask" if name in f.MULTILABEL else target_kind
    target = np.eye(4, dtype=bool)[:2] if target_kind == "mask" else np.zeros(2, np.int64)
    if target_kind == "teacher":
        target = z
    for kind in kinds:
        result = interfaces.c_loss_evaluate(
            interfaces.C_KINDS[kind], "float64", z, target_kind,
            target,
            **interfaces.c_parameters(kind, {"k": 2, "alpha": 1.0}), **{option: value},
        )  # fmt: skip
        rejected = result.message == f"{kind} takes no parameter {option}"
        assert rejected != takes, (kind, option, result.message)
    loss = libsdca.LOSSES[name](**({"k": 2} if name in f.TOPK else {}))
    try:
        libsdca.loss_and_grad(z, target, loss, **{option: value})
        numpy_takes = True
    except TypeError as error:
        numpy_takes = str(error) != f"{name} takes no parameter {option}"
    assert numpy_takes == takes
    for framework in FRAMEWORKS[:2]:
        if importlib.util.find_spec(framework) is None:
            continue
        module = importlib.import_module(f"libsdca.{framework}")
        function = getattr(module, name, None) or getattr(module, f"{name}_loss")
        signature = inspect.signature(function).parameters
        assert (option in signature) == takes, (framework, name, option)

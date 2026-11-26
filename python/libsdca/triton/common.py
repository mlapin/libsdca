"""What the loss families share on the host: the call (the arguments of the row kernel of
libsdca::loss and libsdca_loss) and the launch plans that `launch_torch` and `launch_jax` both
execute.

A plan names its arrays. Inputs: Z (scores), Y (labels, mask, probabilities or teacher
logits), W (class weights, (groups, C)), E (label smoothing, (groups,)), R (row_upstream: one
factor per row, or one for every row), D (the divisor of a mean) and S (statistics a forward
saved). Results: V (per-row values), L (their sum or mean: reduction), D (a mean's divisor,
where the reduction computes it), S (saved statistics) and G (the gradient times upstream * row_upstream / D,
which `launch_torch` may write into Z). Internal arrays (row statistics) are float64. Each
launch writes only arrays it creates, so JAX runs each as one call; a launch whose argument is
absent passes Z, which the kernel does not read.

Kernels convert a scalar argument with tl.cast(x, dtype), not x.to(dtype) or another method:
jax-triton compiles every scalar argument equal to 1 as a constexpr (a Python int), whatever
do_not_specialize says. For a tensor both give the same code.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Any

import triton

from .._frontend import CUDA_MAX_ROWS as ROWS
from .._frontend import KINDS
from .math import bits

CROSS_ENTROPY, TOPK_CROSS_ENTROPY = KINDS["cross_entropy"], KINDS["topk_cross_entropy"]
MULTICLASS_HINGE, MULTILABEL_HINGE = KINDS["multiclass_hinge"], KINDS["multilabel_hinge"]
TOPK_HINGE_ALPHA, TOPK_HINGE_BETA = KINDS["topk_hinge_alpha"], KINDS["topk_hinge_beta"]
MULTILABEL_CROSS_ENTROPY = KINDS["multilabel_cross_entropy"]
PROXIMAL_CROSS_ENTROPY = KINDS["proximal_cross_entropy"]
DISTILLATION, PROXIMAL_DISTILLATION = KINDS["distillation"], KINDS["proximal_distillation"]
# Call.rowwise: no row_upstream, one factor per row, one factor for every row.
UNIFORM, PER_ROW, SCALAR = 0, 1, 2
# Call.saved: no saved statistics, a forward that saves them, a gradient that reads them.
SAVE, SAVED = 1, 2
# Call.reduction: per-row values, their sum, their mean.
SUM, MEAN = 1, 2
# Rows of the kernels (int32 grids) and classes (backward grids of 65535 column blocks).
CLASSES = 2**25


@dataclass(frozen=True)
class Call:
    """A row-kernel call: the loss (`kind`, SDCA_LOSS_*), its parameters, the arrays' shapes
    and dtype names ("float32", "int64", "bool", ...), and what to compute.

    group: rows per weight and smoothing group (0: one group); precision: 0 the storage's
    work, 1 float32, 2 float64; rowwise: UNIFORM, PER_ROW or SCALAR row_upstream (any float
    dtype); divided: the gradient factors are over D[0]; saved: SAVE or SAVED statistics;
    reduction: SUM or MEAN of the values in L (value_dtype), a mean's divisor in D; stride:
    the row stride of row-strided scores (families with `strided`), else 0."""

    kind: int
    n: int
    c: int
    dtype: str
    target: str
    k: int = 1
    gamma: float = 0.0
    temperature: float = 1.0
    ignore_index: int = -100
    ignore: bool = False
    weighted: bool = False
    smoothed: bool = False
    group: int = 0
    precision: int = 0
    upstream: float = 1.0
    rowwise: int = UNIFORM
    divided: bool = False
    saved: int = 0
    reduction: int = 0
    reduction_rows: int = 0  # rows per scalar result under vmap (0: one result)
    values: bool = True
    grad: bool = True
    value_dtype: str = "float64"
    inplace: bool = False
    stride: int = 0  # the scores' row stride in elements (0: c, contiguous rows)

    @classmethod
    def of(
        cls, n: int, c: int, dtype: str, target: str, kind: int, k: int, gamma: float,
        temperature: float, label_smoothing: float, ignore_index: int, ignore: bool,
        precision: int, **options: Any,
    ) -> Call:  # fmt: skip
        """The call of a row kernel's options (the fields of _frontend.Rows): label smoothing
        only says whether cross_entropy reads E."""
        smoothed = kind == CROSS_ENTROPY and label_smoothing != 0
        return cls(
            kind, n, c, dtype, target, k=k, gamma=gamma, temperature=temperature,
            ignore_index=ignore_index, ignore=ignore, smoothed=smoothed, precision=precision,
            **options,
        )  # fmt: skip

    @property
    def fp32(self) -> bool:
        """Float32 work: never for float64 scores."""
        return self.precision != 2 and self.dtype != "float64"

    @property
    def sz(self) -> int:
        """The scores' row stride."""
        return self.stride or self.c

    @property
    def sg(self) -> int:
        """The gradient's row stride: the scores' where it overwrites them, else c."""
        return self.sz if self.inplace else self.c

    @property
    def grad_dtype(self) -> str:
        """The gradient's dtype: the scores' (rounded once, after the upstream factors)."""
        return self.dtype

    @property
    def dense(self) -> bool:
        """Probabilities or teacher logits."""
        return self.target.startswith(("float", "bfloat"))

    @property
    def mask(self) -> bool:
        return self.target in ("bool", "uint8")

    @property
    def indexed_hinge(self) -> bool:
        """Unweighted unsmoothed classifier rows eligible to encode a unique competitor (in
        the float64 statistics, so for every storage dtype with float32 work)."""
        return (
            self.kind == MULTICLASS_HINGE
            and self.fp32
            and self.gamma == 0
            and not self.weighted
            and not self.ignore
        )

    @property
    def short_bfloat_distillation(self) -> bool:
        from . import DISTILLATION_REGISTER_CLASSES

        return (
            self.kind == DISTILLATION
            and self.fp32
            and self.dtype == self.target == "bfloat16"
            and self.c <= DISTILLATION_REGISTER_CLASSES
        )

    @property
    def register_smooth_hinge(self) -> bool:
        """One register tile of storage up to FP32, with bounded positive smoothing."""
        return (
            self.kind == MULTICLASS_HINGE
            and 2 <= self.c <= 1024
            and self.dtype in ("float32", "float16", "bfloat16")
            and self.fp32
            and 2.0**-126 <= self.gamma <= 2.0**127
        )

    @property
    def rows_per_group(self) -> int:
        return self.group or max(1, self.n)

    @property
    def counted(self) -> bool:
        """A mean over the label weights (class weights or an ignored label), not the rows."""
        return (
            self.reduction == MEAN and self.target.startswith(("int", "uint")) and self.target != "uint8"
            and (self.weighted or self.ignore)
        )  # fmt: skip


@dataclass(frozen=True)
class Ref:
    """An array argument of a launch, by name."""

    name: str


@dataclass(frozen=True)
class Array:
    """An array a plan creates; `internal` float64 arrays may travel as 32-bit words."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    internal: bool = True


@dataclass(frozen=True)
class Launch:
    """One kernel launch: arguments by parameter name, constexprs and options (num_warps, ...)
    in meta, and the parameters whose arrays it writes."""

    kernel: Any
    grid: tuple[int, ...]
    args: tuple[tuple[str, Any], ...]
    meta: tuple[tuple[str, Any], ...]
    writes: tuple[str, ...]
    state: tuple[str, ...] = ()  # Explicit read/write outputs shared between partial writers.


@dataclass(frozen=True)
class Plan:
    arrays: tuple[Array, ...]
    launches: tuple[Launch, ...]


@dataclass
class Builder:
    """Collects a plan's arrays and launches."""

    call: Call
    arrays: list[Array] = field(default_factory=list)
    launches: list[Launch] = field(default_factory=list)
    partials: int = 0  # programs whose value sums (array T) the reduction adds up
    reduced: bool = False  # the family's launches write the reduction L themselves

    def drop(self, name: str) -> None:
        """Remove an array no launch will use."""
        self.arrays = [a for a in self.arrays if a.name != name]

    def array(self, name: str, shape: tuple[int, ...], dtype: str, internal: bool = True) -> Ref:
        self.arrays.append(Array(name, shape, dtype, internal))
        return Ref(name)

    def results(self) -> dict[str, Ref]:
        """V and G (or Z, unread, where the call wants none); a reduction's per-row values are
        an internal V."""
        c = self.call
        if not c.values:
            v = Ref("Z")
        elif c.reduction:
            v = self.array("V", (c.n,), "float64")
        else:
            v = self.array("V", (c.n,), c.value_dtype, internal=False)
        g = (
            Ref("Z")
            if c.inplace or not c.grad
            else self.array("G", (c.n, c.c), c.grad_dtype, internal=False)
        )
        return dict(V=v, G=g)

    def inputs(self) -> dict[str, Any]:
        """The arrays and scalars every kernel takes."""
        c = self.call
        return dict(
            Z=Ref("Z"),
            Y=Ref("Y"),
            W=Ref("W" if c.weighted else "Z"),
            E=Ref("E" if c.smoothed else "Z"),
            R=Ref("R" if c.rowwise else "Z"),
            D=Ref("D" if c.divided else "Z"),
            N=c.n,
            GROUP=c.rows_per_group,
            IGNORE=c.ignore_index,
            UP=bits(c.upstream),
        )

    def flags(self) -> dict[str, Any]:
        c = self.call
        return dict(
            C=c.c,
            FP32=c.fp32,
            WEIGHTED=c.weighted,
            IGNORED=c.ignore,
            ROWWISE=c.rowwise,
            DIVIDED=c.divided,
            VALUES=c.values,
            GRAD=c.grad,
            SZ=c.sz,
            SG=c.sg,
        )

    def launch(
        self,
        kernel: Any,
        grid: tuple[int, ...],
        args: dict[str, Any],
        warps: int,
        *,
        state: tuple[str, ...] = (),
        **meta: Any,
    ) -> None:
        """Add a launch with the arguments and constexprs its kernel takes."""
        params = dict(parameters(kernel))
        used = {v.name for launch in self.launches for _, v in launch.args if isinstance(v, Ref)}
        mine = {a.name for a in self.arrays} - used
        args = {k: v for k, v in args.items() if k in params and not params[k]}
        options = _cuda_options()
        meta = {
            k: v
            for k, v in meta.items()
            if params.get(k, k.islower() and (options is None or k in options))
        }
        writes = tuple(k for k, v in args.items() if isinstance(v, Ref) and v.name in mine)
        meta["num_warps"] = warps
        self.launches.append(
            Launch(kernel, grid, tuple(args.items()), tuple(sorted(meta.items())), writes, state)
        )

    def stats(self, columns: int) -> Ref:
        """The row statistics S between a family's statistics and gradient launches: saved
        by a forward (SAVE: a result), read from the forward (SAVED: an input) or internal."""
        if self.call.saved == SAVED:
            return Ref("S")
        return self.array("S", (self.call.n, columns), "float64", self.call.saved != SAVE)

    def plan(self) -> Plan:
        if self.call.reduction and self.call.values and not self.reduced:
            from . import reduce

            reduce.add(self, partials=self.partials)
        return Plan(tuple(self.arrays), tuple(self.launches))


@functools.cache
def _cuda_options() -> frozenset[str] | None:
    """The compile options of this Triton's CUDA backend (None: unknown). Before
    enable_reflect_ftz (Triton 3.5 has none) libdevice flushes subnormal results."""
    try:
        from importlib import import_module

        CUDAOptions = import_module("triton.backends.nvidia.compiler").CUDAOptions
    except ImportError:
        return None
    return frozenset(f.name for f in dataclasses.fields(CUDAOptions))


def parameters(kernel: Any) -> list[tuple[str, bool]]:
    """A kernel's parameters and whether each is a constexpr (also under TRITON_INTERPRET)."""
    if hasattr(kernel, "params"):
        return [(p.name, p.is_constexpr) for p in kernel.params]
    signature = inspect.signature(kernel.fn).parameters
    return [(name, "constexpr" in str(p.annotation)) for name, p in signature.items()]


def pow2(x: int) -> int:
    return triton.next_power_of_2(max(1, x))


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


DTYPES = ("float16", "bfloat16", "float32", "float64")
# Triton 3.2 aborts compiling float64 to bfloat16 stores.
TRITON = (3, 3)


def triton_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", triton.__version__)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def native_bfloat_conversion(cuda: Any) -> bool:
    """Probe the active CUDA device for native BF16; older APIs may report emulation."""
    supported = getattr(cuda, "is_bf16_supported", None)
    if supported is None or "including_emulation" not in inspect.signature(supported).parameters:
        return False
    return supported(including_emulation=False)


def direct_launch(metadata: Any) -> bool:
    """Whether a CUDA driver launch with unused scratch pointers implements this kernel."""
    return (
        metadata.num_ctas == 1
        and math.prod(getattr(metadata, "cluster_dims", None) or (1,)) == 1
        and not any(
            getattr(metadata, flag, 0)
            for flag in (
                "global_scratch_size",
                "profile_scratch_size",
                "launch_cooperative_grid",
                "launch_pdl",
                "instrumentation_mode",
            )
        )
    )


def triton_available() -> str | None:
    """None where this Triton compiles the kernels, else why not."""
    if triton_version() < TRITON:
        return f"Triton {triton.__version__} predates 3.3, which the kernels need"
    return None


def _family(call: Call) -> Any:
    from . import ce, distill, multilabel, prox, topk

    kind = call.kind
    # topk_cross_entropy at k = 1 is cross_entropy.
    if kind in (CROSS_ENTROPY, MULTILABEL_CROSS_ENTROPY) or (
        kind == TOPK_CROSS_ENTROPY and call.k == 1
    ):
        return ce
    if kind in (TOPK_CROSS_ENTROPY, MULTICLASS_HINGE, TOPK_HINGE_ALPHA, TOPK_HINGE_BETA):
        return topk
    if kind == MULTILABEL_HINGE:
        return multilabel
    if kind == PROXIMAL_CROSS_ENTROPY:
        return prox
    if kind in (DISTILLATION, PROXIMAL_DISTILLATION):
        return distill
    raise ValueError(f"unknown loss kind {kind}")


def covers(kind: int, k: int, n: int, classes: int, dtype: str | None) -> str | None:
    """None where the kernels compute a call, else why not."""
    if dtype not in DTYPES:
        return f"the CUDA kernels take {', '.join(DTYPES)} scores, not {dtype}"
    if n > ROWS or classes > CLASSES:
        return f"the CUDA kernels take at most {ROWS} rows of {CLASSES} classes"
    if classes < 2:
        return "the CUDA kernels take at least two classes"
    call = Call(kind, n, classes, dtype, "int64", k=k)
    family = _family(call)
    return family.covers(call) if hasattr(family, "covers") else None


def strided(call: Call) -> bool:
    """Whether the kernels of a call take row-strided scores (Call.stride)."""
    return bool(getattr(_family(call), "STRIDED", False))


def plan(call: Call) -> Plan:
    """The launches of a call."""
    return _family(call).plan(call)


def describe(call: Call) -> str:
    """The arithmetic of a call's kernels, for explain_backend."""
    family = _family(call)
    if hasattr(family, "describe"):
        return family.describe(call)
    return "float32 work" if call.fp32 else "float64"

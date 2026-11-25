"""The array namespace the composed losses are written against.

Every function acts on the last axis of (rows, classes) arrays; reductions keep it with
length one. xp_torch and xp_jax implement it for PyTorch tensors and JAX arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import numpy as np

Array = Any
Carry = tuple[Array, ...]
State = TypeVar("State")

FIRST_ORDER = (
    "This loss has first derivatives only: double backward is not supported. Only "
    "cross_entropy, multilabel_cross_entropy, topk_cross_entropy at k = 1, distillation and "
    "the nonsmooth hinges have higher derivatives"
)


def cpu_labels(label: np.ndarray, classes: int) -> np.ndarray:
    """Build the class indices for an eager CPU gather, reporting its first invalid label."""
    try:
        return np.ravel_multi_index((label,), (classes,))
    except ValueError:
        bad = label[(label < 0) | (label >= classes)]
        if bad.size:
            raise IndexError(f"Target {int(bad[0])} is out of bounds.") from None
        raise


class Namespace(Protocol):
    name: str
    float32: Any
    float64: Any
    int_: Any

    def finfo(self, dtype: Any, /) -> Any: ...
    def work(self, x: Array, /) -> Array:
        """x in its work dtype: float32 for 16-bit floating point, else as given."""
        ...

    def wide(self) -> Any:
        """float64 where the framework computes it, else float32 (JAX without x64)."""
        ...

    def astype(self, x: Array, dtype: Any, /) -> Array: ...
    def result_type(self, a: Array, b: Array, /) -> Any: ...
    def full_like(self, x: Array, value: float, /, dtype: Any = None) -> Array: ...
    def eager_labels(self, label: Array, scores: Array, /) -> Array | None: ...
    def arange(self, n: int, like: Array, /) -> Array:
        """Integer indices (n,) on the device of like."""
        ...

    def where(self, c: Array, a: Array | float, b: Array | float, /) -> Array: ...
    def exp(self, x: Array, /) -> Array: ...
    def expm1(self, x: Array, /) -> Array: ...
    def log(self, x: Array, /) -> Array: ...
    def log1p(self, x: Array, /) -> Array: ...
    def abs(self, x: Array, /) -> Array: ...
    def isfinite(self, x: Array, /) -> Array: ...
    def isnan(self, x: Array, /) -> Array: ...
    def maximum(self, a: Array, b: Array | float, /) -> Array: ...
    def minimum(self, a: Array, b: Array | float, /) -> Array: ...
    def clip(self, x: Array, lo: Array | float, hi: Array | float, /) -> Array: ...
    def nextafter(self, x: Array, toward: Array, /) -> Array: ...
    def bits(self, x: Array, /) -> Array:
        """The bits of float32 or float64 x as signed integers of its width."""
        ...

    def from_bits(self, bits: Array, like: Array, /) -> Array:
        """The floats of like's dtype with these bits (the inverse of bits)."""
        ...

    def sum(self, x: Array, /) -> Array: ...
    def max(self, x: Array, /) -> Array: ...
    def min(self, x: Array, /) -> Array: ...
    def argmax(self, x: Array, /) -> Array:
        """The first index of the row maximum (booleans too)."""
        ...

    def argmin(self, x: Array, /) -> Array: ...
    def count(self, mask: Array, /) -> Array:
        """Integer row counts of a boolean mask."""
        ...

    def all(self, mask: Array, /) -> Array: ...
    def logsumexp(self, x: Array, /) -> Array:
        """log sum exp, differentiable at any order, with the derivative softmax(x)."""
        ...

    def softmax(self, x: Array, /) -> Array:
        """exp(x - logsumexp(x)), as the derivative of logsumexp computes it: the same
        logits give the same probabilities bit for bit."""
        ...

    def log_softmax(self, x: Array, /) -> Array:
        """The framework's fused log-softmax (its own -inf rule: masked entries stay -inf)."""
        ...

    def sort(self, x: Array, /) -> Array:
        """Row values in descending order."""
        ...

    def topk(self, x: Array, k: int, /, *, stable: bool = True) -> tuple[Array, Array]:
        """The k largest values of each row, descending, and their indices. Stable ties prefer the
        lowest index. Differentiable in the values."""
        ...

    def mean_margin(self, best: Array, label: Array, /) -> tuple[Array, Array]:
        """Compensated mean hinge margin and its signed sum before division."""
        ...

    def take(self, x: Array, index: Array, /) -> Array:
        """x[i, index[i, j]], broadcasting the rows of x and index."""
        ...

    def put(self, x: Array, index: Array, values: Array | float, /) -> Array:
        """x with x[i, index[i, j]] = values[i, j] (or a scalar)."""
        ...

    def concat(self, xs: Sequence[Array], /) -> Array: ...
    def pad(self, x: Array, size: int, /) -> Array:
        """x padded with zeros to size classes."""
        ...

    def stop(self, x: Array, /) -> Array:
        """x without derivatives."""
        ...

    def attach(self, value: Array, z: Array, gradient: Array, /) -> Array:
        """value (rows,) whose derivative in z (rows, classes) is gradient, in reverse and
        forward mode; differentiating that derivative raises RuntimeError(FIRST_ORDER).
        Zero gradient entries are +0.0 for either sign of the cotangent."""
        ...

    def attach_smooth(self, value: Array, z: Array, gradient: Array, /) -> Array:
        """value (rows,) whose derivative in z (rows, classes) is gradient, which keeps its own
        derivatives in z: the higher derivatives are those of gradient, at any order and in
        reverse and forward mode. Zero gradient entries are +0.0 for either sign of the
        cotangent."""
        ...

    def target_gradient(
        self, value: Array, target: Array, gradient: Any, scale: float, /, *operands: Any
    ) -> Array:
        """value (the loss of a target: rows, or reduced), unchanged and with its own
        derivatives, whose derivative in target (rows, classes) is
        _targets.cotangent(gradient(xp, *operands), upstream, scale), computed only where
        target is differentiated (first order)."""
        ...

    def grad(self, f: Callable[[Array], Array], x: Array, /) -> Array:
        """The gradient of f(x).sum() in x (a constant of the enclosing derivatives)."""
        ...

    def fori(self, n: int, body: Callable[[int, State], State], state: State, /) -> State: ...
    def while_loop(
        self,
        more: Callable[[Carry], Array],
        body: Callable[[Carry], Carry],
        state: Carry,
        steps: int,
        /,
    ) -> Carry:
        """body applied to state while more(state) (a scalar) holds, at most steps times.
        Where the framework tests more only every few steps (eager CUDA) or not at all
        (torch.func transforms, CUDA graph capture, compiled graphs off the CPU, compiling
        before PyTorch 2.7), body runs more often: it must keep finished rows fixed."""
        ...

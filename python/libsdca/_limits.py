"""The limits of the float work types that thresholds, tolerances and iteration counts derive
from, as include/sdca/core/limits.h: the same definitions, each from the type's digits and
exponent range alone, for the Triton kernels (as tl.constexpr), the composed losses and the
frontend. tests/python/test_limits.py checks them against numpy.finfo, mpmath and the C++
header.

The logarithm of a power of two 2**n is n ln 2, rounded once to the type.
"""

from __future__ import annotations

import decimal
import struct
from typing import Any, NamedTuple

# ln 2 to 50 digits (OEIS A002162).
_LN2 = decimal.Decimal("0.69314718055994530941723212145817656807550013436026")
_CONTEXT = decimal.Context(prec=60)

# The coefficients c_n = (-(n + 1))**n / (n + 1)! of W(y) / y = exp(-W(y)) = sum_n c_n y**n
# (include/sdca/core/lambert.h, kLambertRatio), alternating with terms decreasing as (e y)**n;
# the series' domain y <= 2**LAMBERT_RATIO_LOG2.
LAMBERT_RATIO_LOG2 = -5


def _lambert_ratio(n: int) -> float:
    term = decimal.Decimal((-(n + 1)) ** n)
    for k in range(2, n + 2):
        term = _CONTEXT.divide(term, k)
    return float(term)


LAMBERT_RATIO = tuple(_lambert_ratio(n) for n in range(34))


def _round(value: decimal.Decimal | float, digits: int) -> float:
    """value rounded to float32 (digits 24) or float64 (53)."""
    x = float(value)
    if digits == 24:
        x = struct.unpack("<f", struct.pack("<f", x))[0]
    return x


def _ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


class Limits(NamedTuple):
    """The limits of one work type (fields as the functions of include/sdca/core/limits.h)."""

    digits: int
    min_exponent: int  # as std::numeric_limits: tiny = 2**(min_exponent - 1)
    max_exponent: int  # max < 2**max_exponent
    eps: float  # 2**(1 - digits)
    roundoff: float  # u = eps / 2
    one_below: float  # 1 - u, the largest value below one
    inv_eps: float  # 2**(digits - 1)
    tiny: float  # the smallest normal, 2**(min_exponent - 1)
    max: float
    min_over_eps: float  # the least x with eps x normal
    log_eps: float  # (1 - digits) ln 2
    log_roundoff: float  # -digits ln 2: below it exp(x) < u and W(exp(x)) rounds to exp(x)
    log_min: float  # (min_exponent - 1) ln 2: below it exp(x) is subnormal
    log_denorm_min: float  # (min_exponent - digits) ln 2
    log_max: float  # max_exponent ln 2: above it exp(x) overflows
    log_quarter_max: float  # (max_exponent - 2) ln 2: exp(x) <= max / 4 below it
    exp_reduction_min: float  # (min_exponent - 1/2) ln 2: 2**k exp(r) normal from there on
    exp_reduction_max: float  # (max_exponent - 1/2) ln 2: 2**k exp(r) finite below it
    cbrt_eps: float  # 2**((1 - digits) / 3, truncated): at or above cbrt(eps)
    key_bisection_steps: int  # halvings of the float keys to adjacent values (32, 64)
    value_bisection_steps: int  # halvings of the values (lo + (hi - lo) / 2): 278, 2099
    root_free_steps: int  # free steps of a safeguarded root search (core::root_steps)
    root_steps: int  # free steps, then bisections to an adjacent bracket
    lambert_ratio_terms: int  # terms of the W(y) / y series to below eps / 32 (7, 15)
    lambert_ratio_arg: float  # LAMBERT_RATIO_LOG2 ln 2, the series' largest argument
    prox_tolerance: float  # |sum r| <= prox_tolerance sum |r| (core::softmax::tolerance)

    def log_pow2(self, n: int) -> float:
        """log(2**n) = n ln 2 rounded to the type."""
        return _round(_CONTEXT.multiply(decimal.Decimal(n), _LN2), self.digits)

    def steps_of_order(self, order: int, bits: int) -> int:
        """Steps of an iteration of order `order` from a relative error below 2**-bits to
        below 2**-digits: the bits correct grow as bits * order**n."""
        steps = 0
        while bits < self.digits:
            bits *= order
            steps += 1
        return steps


def _limits(digits: int, min_exponent: int, max_exponent: int) -> Limits:
    def log_pow2(n: int) -> float:
        return _round(_CONTEXT.multiply(decimal.Decimal(n), _LN2), digits)

    def log_pow2_half(m: int) -> float:
        return _round(_CONTEXT.multiply(decimal.Decimal(m) / 2, _LN2), digits)

    eps = 2.0 ** (1 - digits)
    key = digits + _ceil_log2(max_exponent - min_exponent + 2)
    value = max_exponent - min_exponent + digits + 1
    bound = eps / 32
    terms = next(
        n
        for n in range(len(LAMBERT_RATIO))
        if abs(LAMBERT_RATIO[n]) * 2.0 ** (LAMBERT_RATIO_LOG2 * n) <= bound
    )
    return Limits(
        digits=digits,
        min_exponent=min_exponent,
        max_exponent=max_exponent,
        eps=eps,
        roundoff=eps / 2,
        one_below=1 - eps / 2,
        inv_eps=1 / eps,
        tiny=2.0 ** (min_exponent - 1),
        max=(2 - eps) * 2.0 ** (max_exponent - 1),
        min_over_eps=2.0 ** (min_exponent - 1) / eps,
        log_eps=log_pow2(1 - digits),
        log_roundoff=log_pow2(-digits),
        log_min=log_pow2(min_exponent - 1),
        log_denorm_min=log_pow2(min_exponent - digits),
        log_max=log_pow2(max_exponent),
        log_quarter_max=log_pow2(max_exponent - 2),
        exp_reduction_min=log_pow2_half(2 * min_exponent - 1),
        exp_reduction_max=log_pow2_half(2 * max_exponent - 1),
        cbrt_eps=2.0 ** -((digits - 1) // 3),
        key_bisection_steps=key,
        value_bisection_steps=value,
        root_free_steps=key,
        root_steps=2 * key + 1,
        lambert_ratio_terms=terms,
        lambert_ratio_arg=log_pow2(LAMBERT_RATIO_LOG2),
        prox_tolerance=16 * (eps / 2),
    )


FLOAT32 = _limits(24, -125, 128)
FLOAT64 = _limits(53, -1021, 1024)


def limits(dtype: Any) -> Limits:
    """The limits of a float32 or float64 work type: a NumPy, PyTorch, JAX or Triton dtype, or
    its name."""
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    name = str(name).rsplit(".", 1)[-1]
    if name in ("float32", "fp32"):
        return FLOAT32
    if name in ("float64", "fp64"):
        return FLOAT64
    raise TypeError(f"no float32 or float64 work type: {dtype!r}")

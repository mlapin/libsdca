"""The limits of the storage dtypes, from libsdca._limits: float32 and float64 are its work
types, and float16 and bfloat16 take its definitions from their digits and exponent ranges
(as numpy.finfo: digits, min_exponent and max_exponent of std::numeric_limits)."""

from __future__ import annotations

from libsdca import _limits
from libsdca._limits import Limits

LIMITS: dict[str, Limits] = {
    "float16": _limits._limits(11, -13, 16),
    "bfloat16": _limits._limits(8, -125, 128),
    "float32": _limits.FLOAT32,
    "float64": _limits.FLOAT64,
}
EPS: dict[str, float] = {dtype: limits.eps for dtype, limits in LIMITS.items()}

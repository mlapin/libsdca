"""libsdca._limits against numpy.finfo, mpmath and include/sdca/core/limits.h (through
tests/cpp/rewrite_smoke_limits.cpp, where a C++ compiler is available)."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
import pytest
from libsdca import _limits
from libsdca._limits import FLOAT32, FLOAT64, LAMBERT_RATIO, LAMBERT_RATIO_LOG2

ROOT = Path(__file__).resolve().parents[2]
TYPES = [(np.float32, FLOAT32), (np.float64, FLOAT64)]


def _rounded(value: Any, digits: int) -> float:
    """value correctly rounded to `digits` bits (float32 or float64)."""
    with mpmath.workprec(digits):
        return float(+value)


@pytest.mark.parametrize("dtype,lim", TYPES)
def test_finfo(dtype: type, lim: _limits.Limits) -> None:
    info = np.finfo(dtype)
    assert lim.digits == info.nmant + 1
    assert lim.min_exponent - 1 == info.minexp
    assert lim.max_exponent == info.maxexp
    assert lim.eps == float(info.eps)
    assert lim.roundoff == float(info.epsneg) == lim.eps / 2
    assert lim.tiny == float(info.smallest_normal)
    assert lim.max == float(info.max)
    assert dtype(lim.one_below) == np.nextafter(dtype(1), dtype(0))
    assert dtype(lim.inv_eps) + dtype(0.5) == dtype(lim.inv_eps)
    assert dtype(lim.eps) * dtype(lim.min_over_eps) == info.smallest_normal
    assert dtype(lim.eps) * dtype(lim.min_over_eps / 2) < info.smallest_normal


@pytest.mark.parametrize("dtype,lim", TYPES)
def test_logarithms(dtype: type, lim: _limits.Limits) -> None:
    """Each logarithm of a power of two correctly rounded to the type."""
    mpmath.mp.prec = 200
    ln2 = mpmath.log(2)
    for name, n in [
        ("log_eps", 1 - lim.digits),
        ("log_roundoff", -lim.digits),
        ("log_min", lim.min_exponent - 1),
        ("log_denorm_min", lim.min_exponent - lim.digits),
        ("log_max", lim.max_exponent),
        ("log_quarter_max", lim.max_exponent - 2),
        ("lambert_ratio_arg", LAMBERT_RATIO_LOG2),
    ]:
        assert getattr(lim, name) == _rounded(n * ln2, lim.digits), name
        assert lim.log_pow2(n) == getattr(lim, name)
    assert lim.exp_reduction_min == _rounded((2 * lim.min_exponent - 1) * ln2 / 2, lim.digits)
    assert lim.exp_reduction_max == _rounded((2 * lim.max_exponent - 1) * ln2 / 2, lim.digits)
    assert lim.log_max == _rounded(mpmath.log(mpmath.mpf(lim.max)), lim.digits)
    with np.errstate(under="ignore"):
        assert np.exp(dtype(lim.log_min) - dtype(math.log(2))) < lim.tiny
        assert np.exp(dtype(lim.log_denorm_min) - dtype(1)) == 0
        assert np.exp(dtype(lim.log_denorm_min)) > 0


@pytest.mark.parametrize("dtype,lim", TYPES)
def test_counts(dtype: type, lim: _limits.Limits) -> None:
    """The bisection counts close the widest bracket, and the series terms meet eps / 32."""
    finite = 2**lim.digits * (lim.max_exponent - lim.min_exponent + 2)
    assert 2 ** (lim.key_bisection_steps - 1) < finite <= 2**lim.key_bisection_steps
    width = mpmath.mpf(2) ** (lim.max_exponent + 1)
    spacing = mpmath.mpf(2) ** (lim.min_exponent - lim.digits)
    assert lim.value_bisection_steps == int(mpmath.log(width / spacing, 2))
    assert lim.root_steps == lim.root_free_steps + lim.key_bisection_steps + 1
    assert lim.cbrt_eps**3 >= lim.eps and lim.cbrt_eps**3 <= 4 * lim.eps
    y = 2.0**LAMBERT_RATIO_LOG2
    n = lim.lambert_ratio_terms
    assert abs(LAMBERT_RATIO[n]) * y**n <= lim.eps / 32 < abs(LAMBERT_RATIO[n - 1]) * y ** (n - 1)
    assert lim.steps_of_order(3, 1) == math.ceil(math.log(lim.digits, 3))
    assert lim.prox_tolerance == 8 * lim.eps


def test_series() -> None:
    """The coefficients of W(y) / y, and the series against mpmath on its domain."""
    mpmath.mp.prec = 200
    for n, c in enumerate(LAMBERT_RATIO):
        exact = mpmath.mpf(-(n + 1)) ** n / mpmath.factorial(n + 1)
        assert c == float(exact)
    for lim in (FLOAT32, FLOAT64):
        y = mpmath.mpf(2) ** LAMBERT_RATIO_LOG2
        series = sum(
            mpmath.mpf(c) * y**n for n, c in enumerate(LAMBERT_RATIO[: lim.lambert_ratio_terms])
        )
        exact = mpmath.exp(-mpmath.lambertw(y).real)
        assert abs(series / exact - 1) < lim.eps / 16


def test_model_remainder() -> None:
    """|P_8| <= 1.62 on [0, 1] (kModelRemainder of include/sdca/core/project_softmax.h), with
    P_1 = u and P_{k+1} = u P_k + u^2 (u - 1) P_k': sampled at spacing h, the peak is within
    max |P_8'| h / 2 + max |P_8''| h^2 / 4 of the samples' (the sum of the coefficients'
    magnitudes bounds |P_8''|)."""

    def step(p: list[int]) -> list[int]:
        out = [0] * (len(p) + 2)
        for i, c in enumerate(p):
            out[i + 1] += c  # u P
            if i > 0:  # u^2 (u - 1) i c u^(i - 1)
                out[i + 2] += i * c
                out[i + 1] -= i * c
        return out

    p = [0, 1]
    for _ in range(7):
        p = step(p)
    dp = [i * c for i, c in enumerate(p)][1:]
    ddp = [i * c for i, c in enumerate(dp)][1:]

    def value(q: list[int], x: float) -> float:
        return sum(c * x**i for i, c in enumerate(q))

    n = 200000
    peak = max(abs(value(p, i / n)) for i in range(n + 1))
    slope = max(abs(value(dp, i / n)) for i in range(n + 1))
    curvature = sum(abs(c) for c in ddp)
    assert peak + slope / (2 * n) + curvature / (4 * n * n) <= 1.62
    for lim, reach in ((FLOAT32, 2.0**-2), (FLOAT64, 2.0**-6)):
        bound = lim.eps / 32
        assert (
            1.62 * reach**8 / math.factorial(8)
            <= bound
            < 1.62 * (2 * reach) ** 8 / math.factorial(8)
        )


def test_dtype_names() -> None:
    assert _limits.limits(np.float32) is FLOAT32
    assert _limits.limits(np.dtype("float64")) is FLOAT64
    assert _limits.limits("float32") is FLOAT32
    with pytest.raises(TypeError):
        _limits.limits(np.float16)


def _compiler() -> str | None:
    return (
        os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
    )


def test_cpp_header(tmp_path: Path) -> None:
    """The values of include/sdca/core/limits.h (and the counts and constants of the solves)
    equal those of _limits."""
    compiler = _compiler()
    source = ROOT / "tests" / "cpp" / "rewrite_smoke_limits.cpp"
    if compiler is None or not source.exists():
        pytest.skip("needs a C++ compiler and the source tree")
    binary = tmp_path / "limits"
    subprocess.run(
        [compiler, "-std=c++11", "-O1", f"-I{ROOT / 'include'}", str(source), "-o", str(binary)],
        check=True,
    )
    out = subprocess.run([str(binary), "print"], check=True, capture_output=True, text=True).stdout
    values = {"float32": FLOAT32, "float64": FLOAT64}
    seen = 0
    for line in out.splitlines():
        type_name, name, value = line.split()
        expected = getattr(values[type_name], name)
        assert float.fromhex(value) == float(expected), (type_name, name)
        seen += 1
    assert seen == 2 * 21

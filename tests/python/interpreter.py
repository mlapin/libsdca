"""Run fused-kernel checks under Triton's CPU interpreter (TRITON_INTERPRET=1), without a GPU.

The interpreter has no libdevice: importing this module under TRITON_INTERPRET=1 installs
NumPy-based exp, log, log1p, expm1 and sqrt, and lets the device functions of a launch use
the triton.language that the launch patched (Triton patches it again at every device
function call: half of an interpreted run). run() launches every kernel through Triton
(CPU tensors) in a child that imports this module first.
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from typing import Any

import pytest
from child_process import run_python

if os.environ.get("TRITON_INTERPRET") == "1" and importlib.util.find_spec("triton"):
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _exp(x):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
        return tl.exp(x)

    @triton.jit
    def _log(x):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
        return tl.log(x)

    @triton.jit
    def _log1p(x):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
        u = 1 + x
        return tl.where((u == 1) | (u == float("inf")), x, tl.log(u) * (x / (u - 1)))

    @triton.jit
    def _expm1(x):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
        u = tl.exp(x)
        kahan = (u - 1) * (x / tl.log(u))
        return tl.where(u == 1, x, tl.where((u - 1 == -1) | (u == float("inf")), u - 1, kahan))

    @triton.jit
    def _sqrt(x):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
        return tl.sqrt(x)

    libdevice.exp, libdevice.log, libdevice.log1p = _exp, _log, _log1p
    libdevice.expm1, libdevice.sqrt = _expm1, _sqrt

    _interpreter: Any = importlib.import_module("triton.runtime.interpreter")
    if all(hasattr(_interpreter, n) for n in ("_patch_lang", "_LangPatchScope", "GridExecutor")):
        _patch_lang, _call = _interpreter._patch_lang, _interpreter.GridExecutor.__call__
        # The language modules patched by each running launch.
        _patched: list[set[int]] = []

        def _patch_once(fn: Any) -> Any:
            langs = {id(v) for v in fn.__globals__.values() if v is tl or v is tl.core}
            if langs and _patched and langs <= _patched[-1]:
                return _interpreter._LangPatchScope()
            scope = _patch_lang(fn)
            if _patched:
                _patched[-1] |= langs
            return scope

        def _launch(self: Any, *args: Any, **kwargs: Any) -> Any:
            _patched.append(set())
            try:
                return _call(self, *args, **kwargs)
            finally:
                _patched.pop()

        _interpreter._patch_lang = _patch_once
        _interpreter.GridExecutor.__call__ = _launch


def run(script: str, timeout: float = 1200) -> str:
    """Run script in an interpreted child (skip without Triton); return its stdout."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("needs Triton")
    # The interpreter must use CPU tensors even when the parent has a GPU.
    # In particular its float64-to-bfloat16 store is not a hardware conversion.
    env = dict(os.environ, TRITON_INTERPRET="1", SDCA_CUDA_LAUNCHER="off", CUDA_VISIBLE_DEVICES="")
    source = "import interpreter\n" + textwrap.dedent(script)
    result = run_python(
        "-W", "ignore", "-c", source, env=env, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout

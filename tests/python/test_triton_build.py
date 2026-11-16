"""The compiler-less CUDA fallback respects caller-owned toolchain choices."""

import types
from pathlib import Path
from unittest import mock

import pytest
from libsdca import _triton_build


@pytest.mark.parametrize("choice", ["CC", "CXX", "host", "hook", "legacy", "fallback"])
def test_compiler_selection_preserves_user_choices(choice: str) -> None:
    existing = object() if choice == "hook" else None
    knobs = types.SimpleNamespace(build=types.SimpleNamespace(impl=existing))
    triton = types.SimpleNamespace() if choice == "legacy" else types.SimpleNamespace(knobs=knobs)
    environment = {choice: "/chosen/compiler"} if choice in ("CC", "CXX") else {}
    _triton_build.ensure_compiler.cache_clear()
    try:
        with (
            mock.patch.object(_triton_build.sys, "platform", "linux"),
            mock.patch.dict(_triton_build.os.environ, environment, clear=True),
            mock.patch.dict(_triton_build.sys.modules, {"triton": triton}),
            mock.patch.object(
                _triton_build.shutil,
                "which",
                return_value="/usr/bin/gcc" if choice == "host" else None,
            ),
            mock.patch.object(_triton_build.importlib.util, "find_spec", return_value=object()),
        ):
            before = dict(_triton_build.os.environ)
            _triton_build.ensure_compiler()
            assert dict(_triton_build.os.environ) == before
            assert knobs.build.impl is (_triton_build._build if choice == "fallback" else existing)
    finally:
        _triton_build.ensure_compiler.cache_clear()


def test_zig_build_failure_captures_stderr(tmp_path: Path) -> None:
    triton = types.SimpleNamespace(
        knobs=types.SimpleNamespace(build=types.SimpleNamespace(backend_dirs=[]))
    )
    failed = _triton_build.subprocess.CompletedProcess([], 1, "", "error: missing header cuda.h\n")
    with (
        mock.patch.dict(_triton_build.sys.modules, {"triton": triton}),
        mock.patch.object(_triton_build.subprocess, "run", return_value=failed),
        pytest.raises(RuntimeError, match="missing header cuda.h"),
    ):
        _triton_build._build("cuda_utils", str(tmp_path / "u.c"), str(tmp_path), [], [], [])

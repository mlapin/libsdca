"""Reject incompatible native libraries before using their computing entry points, and
mirror the C structs of include/sdca/c_api.h and src/c_api_internal.h."""

import ctypes as ct
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from child_process import run_python
from libsdca._core import (
    _DEFAULTS,
    _Execution,
    _FitOptions,
    _FitResult,
    _Input,
    _InternalOptions,
    _Loss,
    _Targets,
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include/sdca/c_api.h"

# The ctypes mirrors, their C names and SDCA_*_INIT initializers (None: none).
STRUCTS = [
    (_Loss, "sdca_loss", "SDCA_LOSS_INIT"),
    (_Execution, "sdca_execution", "SDCA_EXECUTION_INIT"),
    (_Targets, "sdca_targets", "SDCA_TARGETS_INIT"),
    (_Input, "sdca_input", "SDCA_INPUT_INIT"),
    (_FitOptions, "sdca_fit_options", "SDCA_FIT_OPTIONS_INIT"),
    (_FitResult, "sdca_fit_result", "SDCA_FIT_RESULT_INIT"),
    (_InternalOptions, "sdca_internal_options", None),
]


def _compiler() -> list[str]:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A C compiler is required for this fixture")
    return compiler


class NativeABICompatibilityTests:
    def test_an_older_abi_is_rejected_before_other_calls(self):
        compiler = _compiler()
        # ABI 5 had another set of entry points and structs.
        # Any call beyond the version handshake aborts this fixture's process.
        source = """
            #include <stdlib.h>
            int sdca_abi_version(void) { return 5; }
            const char *sdca_version(void) { return "0.3.1"; }
            const char *sdca_last_error(void) { abort(); }
            int sdca_fit(void) { abort(); }
            int sdca_loss_gradient(void) { abort(); }
            int sdca_update_objective(void) { abort(); }
        """
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            c_source, library = fixture / "abi5.c", fixture / "libabi5.so"
            c_source.write_text(source)
            subprocess.run(
                [*compiler, "-std=c99", "-shared", "-fPIC", str(c_source), "-o", str(library)],
                check=True,
                capture_output=True,
                text=True,
            )
            # Isolate the deliberate aborts so a regressed guard fails this test
            # without terminating the rest of the package suite.
            code = "\n".join(
                [
                    "import sys",
                    "from libsdca._core import _Native",
                    "try:",
                    "    _Native(sys.argv[1])",
                    "except RuntimeError as error:",
                    "    assert 'C ABI' in str(error), str(error)",
                    "else:",
                    "    raise AssertionError('accepted the incompatible ABI 5 library')",
                ]
            )
            run_python("-c", code, str(library), check=True, timeout=30)

    def test_ctypes_structs_follow_the_headers(self):
        if not HEADER.exists():
            pytest.skip("The C headers are in the source tree only")
        compiler = _compiler()
        lines: list[str] = []
        expected: list[float] = []
        for struct, name, init in STRUCTS:
            fields = [field[0] for field in struct._fields_]
            lines.append(f'printf("%zu\\n", sizeof(struct {name}));')
            expected.append(ct.sizeof(struct))
            for field in fields:
                lines.append(f'printf("%zu\\n", offsetof(struct {name}, {field}));')
                expected.append(getattr(struct, field).offset)
            if init is None:
                continue
            # The initializer's values of the scalar fields (the pointers are NULL).
            lines.append(f"{{ struct {name} d = {init};")
            defaults = _DEFAULTS[struct]
            for field, kind in struct._fields_[1:]:
                value = f"(size_t)d.{field}" if kind is ct.c_void_p else f"d.{field}"
                lines.append(f'printf("%.17g\\n", (double){value});')
                expected.append(float(defaults.get(field, 0)))
            lines.append("}")
        source = "\n".join(
            [
                "#include <stddef.h>",
                "#include <stdio.h>",
                '#include "c_api_internal.h"',
                "int main(void) {",
                *lines,
                "  return 0;",
                "}",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            c_source, program = Path(directory) / "layout.c", Path(directory) / "layout"
            c_source.write_text(source)
            include = ["-I", str(ROOT / "include"), "-I", str(ROOT / "src")]
            build = [*compiler, "-std=c99", *include, str(c_source), "-o", str(program)]
            subprocess.run(build, check=True, capture_output=True, text=True)
            out = subprocess.run([str(program)], check=True, capture_output=True, text=True)
        assert [float(v) for v in out.stdout.split()] == expected

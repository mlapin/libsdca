"""Reject incompatible native libraries before using their computing entry points, and
mirror the C structs of include/sdca/c_api.h."""

import ctypes as ct
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from child_process import run_python
from libsdca._core import _DEFAULT_OPTIONS, _FitResult, _Options

HEADER = Path(__file__).resolve().parents[2] / "include/sdca/c_api.h"


def _compiler() -> list[str]:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A C compiler is required for this fixture")
    return compiler


class NativeABICompatibilityTests:
    def test_positional_abi_is_rejected_before_other_calls(self):
        compiler = _compiler()
        # ABI 4 took the options as positional arguments.
        # Any call beyond the version handshake aborts this fixture's process.
        source = """
            #include <stdlib.h>
            int sdca_abi_version(void) { return 4; }
            const char *sdca_version(void) { return "0.3.0"; }
            const char *sdca_last_error(void) { abort(); }
            size_t sdca_history_fields(void) { abort(); }
            int sdca_fit(void) { abort(); }
            int sdca_loss_gradient(void) { abort(); }
            int sdca_loss_gradient_threads(void) { abort(); }
            int sdca_update_objective(void) { abort(); }
            int sdca_update_objective_threads(void) { abort(); }
        """
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            c_source, library = fixture / "abi4.c", fixture / "libabi4.so"
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
                    "from libsdca import Solver",
                    "try:",
                    "    Solver(library=sys.argv[1])",
                    "except RuntimeError as error:",
                    "    assert 'C ABI' in str(error), str(error)",
                    "else:",
                    "    raise AssertionError('accepted the incompatible ABI 4 library')",
                ]
            )
            run_python("-c", code, str(library), check=True, timeout=30)

    def test_ctypes_structs_follow_the_header(self):
        if not HEADER.exists():
            pytest.skip("The C header is in the source tree only")
        compiler = _compiler()
        fields = [field[0] for field in _Options._fields_]
        result = [field[0] for field in _FitResult._fields_]
        # Offsets and sizes of both structs, then the values of SDCA_OPTIONS_DEFAULT.
        lines = [f"O(sdca_options, {name});" for name in fields]
        lines += [f"O(sdca_fit_result, {name});" for name in result]
        lines += [f'printf("%.17g\\n", (double)d.{name});' for name in fields[1:]]
        source = "\n".join(
            [
                "#include <stddef.h>",
                "#include <stdio.h>",
                '#include "sdca/c_api.h"',
                '#define O(s, f) printf("%zu\\n", offsetof(struct s, f))',
                "int main(void) {",
                "  struct sdca_options d = SDCA_OPTIONS_DEFAULT;",
                '  printf("%zu\\n%zu\\n", sizeof(struct sdca_options), sizeof(struct sdca_fit_result));',
                *lines,
                "  return 0;",
                "}",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            c_source, program = Path(directory) / "layout.c", Path(directory) / "layout"
            c_source.write_text(source)
            include = str(HEADER.parents[1])
            build = [*compiler, "-std=c99", "-I", include, str(c_source), "-o", str(program)]
            subprocess.run(build, check=True, capture_output=True, text=True)
            out = subprocess.run([str(program)], check=True, capture_output=True, text=True)
        values = out.stdout.split()
        expected = [ct.sizeof(_Options), ct.sizeof(_FitResult)]
        expected += [getattr(_Options, name).offset for name in fields]
        expected += [getattr(_FitResult, name).offset for name in result]
        assert [int(v) for v in values[: len(expected)]] == expected
        defaults = [float(v) for v in values[len(expected) :]]
        assert defaults == [float(_DEFAULT_OPTIONS[name]) for name in fields[1:]]

"""Optional native C ABI loaded by ctypes; pure tensor installs need no compiler."""

import json
import logging
import shutil
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

# PEP 517 executes this script without necessarily adding its directory to
# sys.path. The helper is shipped in the sdist, not an external build package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdca_build import (
    COMPILE_FLAGS,
    blas_config,
    blas_package,
    enabled,
    use_available_compilers,
    version,
)

HEADERS = sorted(p.relative_to(ROOT / "include") for p in (ROOT / "include/sdca").rglob("*.h"))


def _define(macro: str) -> tuple[str, str]:
    """A macro NAME or NAME=VALUE as (NAME, VALUE), VALUE 1 by default."""
    name, _, value = macro.partition("=")
    return name, value or "1"


extensions = []
variant = "pure"
requirements = ["numpy>=1.23"]
# The BLAS package a PACKAGE build links and requires (docs/API.md#blas), recorded next to the
# library for python/libsdca/_native.py.
package = None
if enabled("SDCA_BUILD_NATIVE"):
    backend, macros, link = blas_config()
    variant = backend.lower()
    if backend == "PACKAGE":
        package = blas_package()
        requirements.append(package["requirement"])
    extensions = [
        Extension(
            "libsdca._libsdca_native",
            ["src/c_api.cpp"],
            include_dirs=["include"],
            depends=[str(Path("include") / h) for h in HEADERS]
            + sorted(str(p) for p in Path("src").glob("*.h")),
            define_macros=[
                *(_define(macro) for macro in macros),
                ("SDCA_USE_CHRONO", "1"),
                ("SDCA_VERSION", f'"{version()}"'),
            ],
            extra_compile_args=[*COMPILE_FLAGS, "-g0"],
            extra_link_args=[*link, "-pthread"],
            language="c++",
        )
    ]


class NativeBuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        # Distutils does not track linker flags or compiler environment in
        # its timestamp cache. Rebuild when producing a new native artifact.
        self.force = True

    def build_extensions(self):
        for change in use_available_compilers(self.compiler):
            self.announce(change, level=logging.WARNING)
        super().build_extensions()
        for path in self._records():
            if package is None:
                path.unlink(missing_ok=True)
            else:
                record = {k: package[k] for k in ("distribution", "module", "library")}
                path.write_text(json.dumps({"blas_package": record}, indent=2) + "\n")

    def _records(self):
        """The records of the BLAS package next to the libraries (python/libsdca/_native.py)."""
        return [
            Path(self.get_ext_fullpath(ext.name)).with_suffix(".json") for ext in self.extensions
        ]

    def get_outputs(self):
        records = [str(path) for path in self._records()] if package is not None else []
        return [*super().get_outputs(), *records]


class NativeBuildPy(build_py):
    """Copies the kernel headers into the wheel (libsdca/include) for the first-use builds
    of the PyTorch and JAX CPU kernels; source and editable trees include include/."""

    def _headers(self):
        base = Path(self.build_lib) / "libsdca" / "include"
        return {str(base / name): ROOT / "include" / name for name in HEADERS}

    def run(self):
        super().run()
        for destination, source in self._headers().items():
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def get_outputs(self, include_bytecode: bool = True):
        outputs = super().get_outputs(include_bytecode)
        return [*outputs, *(name for name in self._headers() if name not in outputs)]

    def get_output_mapping(self):
        outputs = super().get_output_mapping()
        # Strict editable installs construct a link tree from this mapping, rather
        # than copying every file that run() wrote into build_lib.
        for destination, source in self._headers().items():
            outputs[destination] = str(source.resolve())
        return outputs


class NativeWheel(bdist_wheel):
    def get_tag(self):
        # ctypes loads the library: one wheel serves every Python 3 on a platform.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    packages=find_packages("python"),
    package_dir={"": "python"},
    package_data={
        "libsdca": [
            "py.typed",
            "_torch_cpu.cpp",
            "_jax_cpu.cpp",
            "_jax_cuda.cpp",
            "_frontend_cpu.h",
            "_torch_cuda_plan.h",
        ]
    },
    ext_modules=extensions,
    install_requires=requirements,
    cmdclass={"build_ext": NativeBuildExt, "build_py": NativeBuildPy, "bdist_wheel": NativeWheel},
    # Changing backend or switching to pure must not reuse stale native files.
    options={
        "build": {"build_base": "build/package-" + variant},
        # Setuptools otherwise records the account that owns the staging directory.
        "sdist": {"owner": "root", "group": "wheel" if sys.platform == "darwin" else "root"},
    },
)

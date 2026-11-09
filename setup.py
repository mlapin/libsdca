"""Optional native C ABI loaded by ctypes; pure tensor installs need no compiler."""

import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

# PEP 517 executes this script without necessarily adding its directory to
# sys.path. The helper is shipped in the sdist, not an external build package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdca_build import blas_config, enabled

extensions = []
variant = "pure"
if enabled("SDCA_BUILD_NATIVE"):
    backend, macro, link = blas_config()
    variant = backend.lower()
    extensions = [
        Extension(
            "libsdca._libsdca_native",
            ["src/c_api.cpp"],
            include_dirs=["include"],
            depends=sorted(str(p) for p in Path("include").rglob("*") if p.suffix in (".h", ".cpp"))
            + ["src/tensor_losses.h"],
            define_macros=[
                (macro, "1"),
                ("NDEBUG", None),
                ("SDCA_USE_CHRONO", "1"),
                ("SDCA_VERSION", '"0.3.0"'),
            ],
            extra_compile_args=[
                "-std=c++11",
                "-O3",
                "-g0",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fvisibility=hidden",
                "-fvisibility-inlines-hidden",
                "-pthread",
            ],
            # Loss rows run on std::thread workers.
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


class NativeWheel(bdist_wheel):
    def get_tag(self):
        # ctypes loads the library: one wheel serves every Python 3 on a platform.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    packages=["libsdca"],
    package_dir={"": "python"},
    package_data={"libsdca": ["py.typed", "_torch_entropy.cpp", "_torch_*.h", "_jax_ffi.cpp"]},
    ext_modules=extensions,
    cmdclass={"build_ext": NativeBuildExt, "bdist_wheel": NativeWheel},
    # Changing backend or switching to pure must not reuse stale native files.
    options={"build": {"build_base": "build/package-" + variant}},
)

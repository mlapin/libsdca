"""Optional native C ABI loaded by ctypes; pure tensor installs need no compiler."""

import importlib.util
import os
import shutil
import sys
import warnings
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

# PEP 517 executes this script without necessarily adding its directory to
# sys.path. The helper is shipped in the sdist, not an external build package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sdca_build import blas_config, enabled, version

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
                ("SDCA_VERSION", f'"{version()}"'),
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


if sys.platform == "linux" and sys.version_info >= (3, 11) and enabled("SDCA_BUILD_NATIVE"):
    # This handler uses the stable XLA FFI C ABI, not Python or CUDA build libraries.
    # The wheel therefore works without a compiler or a CUDA toolkit on the user's host.
    spec = importlib.util.find_spec("jaxlib")
    if spec is None or not spec.submodule_search_locations:
        if os.environ.get("CIBUILDWHEEL"):
            raise RuntimeError("building the Linux wheel requires jaxlib's XLA FFI headers")
        warnings.warn(
            "jaxlib is not installed; skipping prebuilt _jax_cuda_native (built on first use)",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        ffi_include = Path(next(iter(spec.submodule_search_locations))) / "include"
        extensions.append(
            Extension(
                "libsdca._jax_cuda_native",
                ["python/libsdca/_jax_cuda.cpp"],
                include_dirs=[str(ffi_include)],
                extra_compile_args=[
                    "-std=c++17",
                    "-O2",
                    "-g0",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
                extra_link_args=["-ldl"],
                language="c++",
            )
        )


class NativeBuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        # Distutils does not track linker flags or compiler environment in
        # its timestamp cache. Rebuild when producing a new native artifact.
        self.force = True


class NativeBuildPy(build_py):
    def run(self):
        super().run()
        # Canonical native headers are copied into the wheel for first-use JIT builds.
        # Source/editable builds use the same canonical header through a relative include.
        destination = Path(self.build_lib) / "libsdca" / "_cpu_storage.h"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "include/sdca/math/storage.h", destination)

    def get_outputs(self, include_bytecode: bool = True):
        outputs = super().get_outputs(include_bytecode)
        storage = str(Path(self.build_lib) / "libsdca" / "_cpu_storage.h")
        return outputs if storage in outputs else [*outputs, storage]

    def get_output_mapping(self):
        outputs = super().get_output_mapping()
        storage = str(Path(self.build_lib) / "libsdca" / "_cpu_storage.h")
        # Strict editable installs construct a link tree from this mapping, rather
        # than copying every file that run() wrote into build_lib.
        outputs[storage] = str((ROOT / "include/sdca/math/storage.h").resolve())
        return outputs


class NativeWheel(bdist_wheel):
    def get_tag(self):
        # ctypes loads the library: one wheel serves every Python 3 on a platform.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    packages=["libsdca"],
    package_dir={"": "python"},
    package_data={
        "libsdca": [
            "py.typed",
            "_torch_entropy.cpp",
            "_torch_*.h",
            "_cpu_*.h",
            "_jax_ffi.cpp",
            "_jax_cuda.cpp",
        ]
    },
    ext_modules=extensions,
    cmdclass={"build_ext": NativeBuildExt, "build_py": NativeBuildPy, "bdist_wheel": NativeWheel},
    # Changing backend or switching to pure must not reuse stale native files.
    options={"build": {"build_base": "build/package-" + variant}},
)

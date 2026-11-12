"""Dependency discovery contracts; only standard-library mocks are needed."""

import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def load_module(name: str, path: Path) -> ModuleType:
    """Import the file at path as a module called name."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module("sdca_build_test", Path(__file__).resolve().parents[2] / "sdca_build.py")


class BuildConfigTests:
    @pytest.mark.skipif(
        not importlib.util.find_spec("coverage"), reason="coverage tool is not installed"
    )
    def test_coverage_combines_host_paths_and_preserves_inputs(self):
        from coverage import CoverageData

        tool = load_module(
            "coverage_tool_test", Path(__file__).resolve().parents[2] / "tools/coverage.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cpu"
            package = root / "python/libsdca"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "dispatch.py").write_text(
                "def cpu():\n    return 1\n\ndef gpu():\n    return 2\n"
            )
            (package / "device.py").write_text(
                "import triton\n\n@triton.jit\ndef kernel():\n    return 3\n\ndef launch():\n    return 4\n"
            )
            (root / "pyproject.toml").write_text((tool.ROOT / "pyproject.toml").read_text())
            build = root / "build"
            build.mkdir()
            cpu_path, gpu_path = build / ".coverage", Path(directory) / "gpu.coverage"
            cpu = CoverageData(basename=str(cpu_path))
            cpu.add_lines(
                {str(package / "dispatch.py"): {1, 2, 4}, str(package / "device.py"): {1, 3, 4, 7}}
            )
            cpu.write()
            other = Path(directory) / "gpu/python/libsdca"
            gpu = CoverageData(basename=str(gpu_path))
            gpu.add_lines(
                {str(other / "dispatch.py"): {1, 4, 5}, str(other / "device.py"): {1, 3, 4, 7, 8}}
            )
            gpu.write()
            original = [path.read_bytes() for path in (cpu_path, gpu_path)]
            with patch.object(tool, "ROOT", root), redirect_stdout(io.StringIO()):
                with pytest.raises(ValueError, match="Missing source manifest"):
                    tool.python_report(build, [])
                tool.write_python_manifest(cpu_path)
                tool.write_python_manifest(gpu_path)
                with pytest.raises(subprocess.CalledProcessError) as failure:
                    tool.python_report(build, [])
                assert failure.value.returncode == 2
                with redirect_stdout(io.StringIO()) as message:
                    assert tool.python_report(build, [], collect_only=True) < 90.0
                assert "PROVISIONAL" in message.getvalue()
                assert (build / "python-provisional.json").is_file()
                assert tool.python_report(build, [gpu_path]) == 100.0
                output = (build / ".coverage-host").read_bytes()
                gpu_manifest = Path(str(gpu_path) + ".sources.json")
                saved_manifest = gpu_manifest.read_bytes()
                incompatible = json.loads(saved_manifest)
                incompatible["files"]["dispatch.py"] = "0" * 64
                gpu_manifest.write_text(json.dumps(incompatible))
                with pytest.raises(ValueError, match="Incompatible coverage sources"):
                    tool.python_report(build, [gpu_path])
                assert output == (build / ".coverage-host").read_bytes()
                gpu_manifest.write_bytes(saved_manifest)
                (package / "dispatch.py").write_text("def changed():\n    return 10\n")
                with pytest.raises(ValueError, match="Incompatible coverage sources"):
                    tool.python_report(build, [gpu_path])
                assert output == (build / ".coverage-host").read_bytes()
            assert original == [path.read_bytes() for path in (cpu_path, gpu_path)]
            report = json.loads((build / "python.json").read_text())
            device = next(
                value for name, value in report["files"].items() if name.endswith("device.py")
            )
            assert 5 in device["excluded_lines"]
            assert 8 in device["executed_lines"]
            assert 8 not in device["excluded_lines"]

    def test_python_collection_keeps_cpp_coverage_gate(self):
        tool = load_module(
            "coverage_tool_test", Path(__file__).resolve().parents[2] / "tools/coverage.py"
        )
        with (
            patch.object(
                tool, "run", side_effect=subprocess.CalledProcessError(2, ["gcovr"])
            ) as run,
            patch.object(tool, "python_report") as python_report,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                tool.report(Path("unused"), [], collect_python=True)
            python_report.assert_not_called()
            command = run.call_args.args
            assert command[command.index("--fail-under-line") + 1] == "90"

    def test_quality_versions_cover_changelog_and_c_header(self, subtests: pytest.Subtests):
        root = Path(__file__).resolve().parents[2]
        tool = load_module("quality_tool_test", root / "tools/quality.py")
        tool.check_versions()
        files = [
            "CHANGELOG.md", "CMakeLists.txt", "docs/C.md", "include/sdca/c_api.h",
            "matlab/make.m", "pyproject.toml", "python/build.py", "python/libsdca/__init__.py",
            "rust/Cargo.toml", "setup.py", "CITATION.cff", "python/README.md", "rust/README.md",
        ]  # fmt: skip
        version = tool._version(r'^version = "([^"]+)"', "pyproject.toml")
        heading = tool._version(rf"^(## {re.escape(version)} \(.*\))$", "CHANGELOG.md")
        for path, old, new in [
            ("CHANGELOG.md", heading, "## 9.9.9 (2026-01-01)"),
            ("CHANGELOG.md", heading, f"## {version} (2026-13-01)"),
            ("include/sdca/c_api.h", f'e.g. "{version}"', 'e.g. "9.9.9"'),
            ("CITATION.cff", f"version: {version}", "version: 9.9.9"),
            ("CITATION.cff", "date-released: ", "date-released: 1999-01-01 # "),
            ("python/README.md", f"libsdca/blob/v{version}/", "libsdca/blob/v9.9.9/"),
            ("rust/README.md", f"--tag v{version}", "--tag v9.9.9"),
        ]:
            with subtests.test(path=path, new=new), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory)
                for name in files:
                    (copy / name).parent.mkdir(parents=True, exist_ok=True)
                    (copy / name).write_text((root / name).read_text())
                target = copy / path
                assert old in target.read_text()
                target.write_text(target.read_text().replace(old, new, 1))
                with patch.object(tool, "ROOT", copy), pytest.raises((SystemExit, ValueError)):
                    tool.check_versions()

    def test_installed_runner_rejects_cached_source_import(self, subtests: pytest.Subtests):
        runner = load_module("installed_runner_test", Path(__file__).with_name("conftest.py"))
        for source, installed in [(True, True), (True, False), (False, True)]:
            with (
                subtests.test(source=source, installed=installed),
                tempfile.TemporaryDirectory() as directory,
            ):
                parent = runner.SOURCE if source else Path(directory)
                package = SimpleNamespace(__file__=str(parent / "libsdca" / "__init__.py"))
                with patch.dict(sys.modules, {"libsdca": package}):
                    if source and installed:
                        with pytest.raises(pytest.UsageError, match="source tree"):
                            runner.tested_libsdca(installed=True)
                    else:
                        assert (
                            runner.tested_libsdca(installed=installed)
                            == Path(package.__file__).resolve()
                        )

    def test_boolean_flags_and_invalid_standalone_mode(self, subtests: pytest.Subtests):
        for value, expected in (
            ("ON", True),
            ("true", True),
            ("1", True),
            ("OFF", False),
            ("false", False),
            ("0", False),
        ):
            with subtests.test(value=value), patch.dict(build.os.environ, {"SDCA_TEST": value}):
                assert build.enabled("SDCA_TEST") == expected
        with patch.dict(build.os.environ, {}, clear=True):
            assert build.enabled("SDCA_TEST")
            assert not build.enabled("SDCA_TEST", default=False)
        with patch.dict(build.os.environ, {"SDCA_TEST": "typo"}):
            with pytest.raises(ValueError):
                build.enabled("SDCA_TEST")
        standalone = load_module(
            "standalone_build_test", Path(__file__).resolve().parents[2] / "python/build.py"
        )
        with (
            tempfile.TemporaryDirectory() as out,
            patch.object(standalone.subprocess, "run") as compile_native,
            patch.object(standalone.subprocess, "check_output", return_value="g++ 0\n"),
        ):
            lib = standalone.build("BUILTIN", out)
            command = compile_native.call_args.args[0]
        # One library with the strict IEEE flags of the wheel.
        assert lib.name == "libsdca_c.so"
        assert {"-O3", "-DNDEBUG", "-fno-fast-math", "-ffp-contract=off"} <= set(command)
        assert not any("fast-math" in f and f != "-fno-fast-math" for f in command)
        assert "-Ofast" not in command

    def test_external_candidates(self, subtests: pytest.Subtests):
        for broken in [None, OSError("missing library"), AttributeError("no CBLAS")]:
            with (
                subtests.test(broken=broken),
                patch.dict(build.os.environ, {}, clear=True),
                patch.object(
                    build.ctypes.util,
                    "find_library",
                    side_effect=["libblas.so.3", "/opt/libopenblas.so"],
                ),
                patch.object(build.ctypes, "CDLL") as load,
                patch.object(build.os.path, "exists", return_value=True),
            ):
                if broken is not None:
                    load.side_effect = [broken, Mock()]
                result = build.blas_config("EXTERNAL")
                assert result == (
                    "EXTERNAL",
                    "BLAS_DEFAULT_LOCAL_HEADER",
                    ["-l:libblas.so.3"] if broken is None else ["/opt/libopenblas.so"],
                )

    def test_shared_cache_library_links_by_name(self):
        # macOS 11+ finds /usr/lib/libblas.dylib in the dyld shared cache, not on disk.
        with (
            patch.dict(build.os.environ, {}, clear=True),
            patch.object(build.ctypes.util, "find_library", return_value="/usr/lib/libblas.dylib"),
            patch.object(build.ctypes, "CDLL"),
            patch.object(build.os.path, "exists", return_value=False),
        ):
            assert build.blas_config("EXTERNAL") == (
                "EXTERNAL",
                "BLAS_DEFAULT_LOCAL_HEADER",
                ["-lblas"],
            )

    def test_missing_and_explicit_backends(self):
        with (
            patch.dict(build.os.environ, {}, clear=True),
            patch.object(build.ctypes.util, "find_library", return_value=None) as find,
        ):
            assert build.blas_config("BUILTIN") == ("BUILTIN", "BLAS_BUILTIN", [])
            find.assert_not_called()
            assert build.blas_config("AUTO") == ("BUILTIN", "BLAS_BUILTIN", [])
            with pytest.raises(RuntimeError):
                build.blas_config("EXTERNAL")
            with pytest.raises(ValueError):
                build.blas_config("typo")
        with patch.dict(build.os.environ, {"SDCA_BLAS": "-L/custom -lopenblas"}, clear=True):
            assert build.blas_config("AUTO")[2] == ["-L/custom", "-lopenblas"]

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch") or shutil.which("c++") is None,
        reason="PyTorch or a C++ compiler is not installed",
    )
    def test_cpu_extension_compiles_for_generic_cpus(self):
        # The flags of arm64 and x86 without AVX2: no CI runner builds the extension there.
        tool = load_module(
            "extension_builds_test",
            Path(__file__).resolve().parents[2] / "tools/check_extension_builds.py",
        )
        label, status, output = tool.compile_once("c++", "DEFAULT")
        assert status == 0, f"{label}:\n{output}"

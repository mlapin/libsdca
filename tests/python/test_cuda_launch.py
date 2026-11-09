"""The fused CUDA paths' launcher gate, diagnostics, batch-size reuse and large batches."""

import ast
import contextlib
import dataclasses
import importlib.util
import math
import os
import sys
import textwrap
import time
import types
import warnings
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import interpreter
import pytest
import torch
import torch.nn.functional as F

# Imported before the tests swap sys.modules, so their patches of load reach the launcher.
import torch.utils.cpp_extension  # noqa: F401
from child_process import run_python
from libsdca import _cpu_entropy, _cuda_launch, _extensions
from libsdca.torch import explain_backend, sdca_loss

# A nonsmooth and a smoothed hinge: (loss, gamma) of _case.
HINGES = ("topk_hinge_alpha", "topk_hinge_beta_smooth")


def _case(loss: str, gamma: float = 0.5) -> tuple[str, float]:
    """The loss name and gamma of a test case: gamma smooths only a _smooth case."""
    return loss.removesuffix("_smooth"), gamma if loss.endswith("_smooth") else 0.0


def _triton():
    try:
        import triton
        from libsdca import _triton_entropy, _triton_hinge, _triton_reduce
    except ImportError:
        return None
    return triton, _triton_entropy, _triton_hinge, _triton_reduce


def _loaded_triton() -> tuple[types.ModuleType, ...]:
    """_triton() where the test class already requires Triton."""
    modules = _triton()
    assert modules is not None
    return modules


@contextlib.contextmanager
def _setting(value: str) -> Iterator[None]:
    """SDCA_CUDA_LAUNCHER=value, reread by the launcher status."""
    with mock.patch.dict(os.environ, {_cuda_launch.ENV: value}):
        _cuda_launch.status.cache_clear()
        try:
            yield
        finally:
            _cuda_launch.status.cache_clear()


def _variants(kernel: object) -> int:
    """Compiled variants in Triton's in-memory cache of a JIT function."""
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(cache[0]) for cache in caches.values())
    return sum(len(cache) for cache in getattr(kernel, "cache", {}).values())


def _reference(
    z: torch.Tensor, y: torch.Tensor, loss: str, reduction: str, gamma: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Values and gradients from float64 composed or boolean-mask tensor losses."""
    x = z.detach().double().requires_grad_()
    if loss == "cross_entropy":
        value = F.cross_entropy(x, y, reduction=reduction)
    else:
        mask = F.one_hot(y, z.shape[1]).bool()
        loss, gamma = _case(loss, gamma)
        value = sdca_loss(x, mask, loss, backend="torch", gamma=gamma, reduction=reduction)
    value.backward(torch.full_like(value, 0.75))
    assert x.grad is not None
    return value.detach(), x.grad


# _triton_entropy._LAYOUT and _triton_hinge._LAYOUT: VALUE, LOG, BAD, STATUS row offsets.
LAYOUTS = {"cross_entropy": (4, 5, 6, 7), "hinge": (0, -1, 3, 4)}
MAX = sys.float_info.max


def reduction_cases(n: int) -> Iterator[tuple[str, list[float], list[float], int | None]]:
    """(name, row values, their logs, invalid row) for the reduction alone."""
    g = torch.Generator().manual_seed(n)
    ordinary = (3 * torch.rand(n, generator=g, dtype=torch.float64) + 1e-3).tolist()
    yield "ordinary", ordinary, [math.log(v) for v in ordinary], None
    yield "sum overflows", [MAX / 4] * n, [math.log(MAX / 4)] * n, None
    yield "tiny", [1e-310] * n, [math.log(1e-310)] * n, None
    unrepresentable = [math.log(2)] * n
    logs = [math.log(math.log(2))] * n
    unrepresentable[n - 2 if n > 1 else 0] = math.inf
    # A loss of e * MAX keeps every mean away from the overflow boundary.
    logs[n - 2 if n > 1 else 0] = math.log(MAX) + 1.0
    yield "unrepresentable", unrepresentable, logs, None
    yield "invalid last row", ordinary, [math.log(v) for v in ordinary], n - 1


def reduction_reference(
    values: list[float], logs: list[float], reduction: str, log_domain: bool
) -> float:
    """The reduction in Python floats: exact sums, a scaled mean, log-sum-exp tails."""
    n, maximum = len(values), max(values)
    if log_domain and not (sys.float_info.min <= maximum < math.inf):
        log_max = max(logs)
        if log_max == -math.inf:
            return 0.0
        total = log_max + math.log(math.fsum(math.exp(v - log_max) for v in logs))
        exponent = total - (math.log(n) if reduction == "mean" else 0.0)
        try:
            return math.exp(exponent)
        except OverflowError:
            return math.inf
    total = sum(values) if maximum * n == math.inf else math.fsum(values)
    if reduction == "sum":
        return total
    if total == math.inf and maximum < math.inf:
        return math.fsum(v / maximum for v in values) / n * maximum
    return total / n


def check_reduction(reduce: types.ModuleType, device: str, sizes: list[int]) -> None:
    """_triton_reduce.launch on synthetic row statistics against reduction_reference."""
    codes = {"none": 0, "mean": 1, "sum": 2}
    for kind, layout in LAYOUTS.items():
        value, log, bad, status = layout
        for n in sizes:
            for name, values, logs, invalid in reduction_cases(n):
                if kind == "hinge" and name in ("tiny", "unrepresentable"):
                    continue
                for reduction, code in codes.items():
                    stats = torch.zeros(reduce.size(n, status), dtype=torch.float64)
                    stats[value * n : (value + 1) * n] = torch.tensor(values, dtype=torch.float64)
                    if log >= 0:
                        stats[log * n : (log + 1) * n] = torch.tensor(logs, dtype=torch.float64)
                    if invalid is not None:
                        stats[bad * n + invalid] = 1.0
                    stats = stats.to(device)
                    out = torch.empty((n,) if code == 0 else (), dtype=torch.float64, device=device)
                    reduce.launch(stats, out, n, code, layout)
                    message = f"{kind} {name} n={n} {reduction}"
                    assert stats[status * n].item() == float(invalid is not None), message
                    if code == 0:
                        assert out.tolist() == values, message
                        continue
                    expected = reduction_reference(values, logs, reduction, log >= 0)
                    actual = out.item()
                    if math.isinf(expected):
                        assert actual == expected, message
                    else:
                        # Subnormal results keep fewer digits.
                        rtol = 1e-11 if abs(expected) < 1e-300 else 1e-12
                        assert abs(actual - expected) <= rtol * abs(expected), message


def _fused(
    z: torch.Tensor, y: torch.Tensor, loss: str, reduction: str, gamma: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z.detach().clone().requires_grad_()
    loss, gamma = _case(loss, gamma)
    value = sdca_loss(x, y, loss, gamma=gamma, reduction=reduction)
    value.backward(torch.full_like(value, 0.75))
    assert x.grad is not None
    return value.detach(), x.grad


class LauncherGateTests:
    def test_version_gate(self):
        for version in ("3.3.1", "3.5.1", "3.8.0", "3.8.0+git1a2b3c", "3.9.0"):
            assert _cuda_launch.version_gate(version) is None, version
        for version in ("2.3.1", "3.2.0", "3.10.0", "4.0.0", "unknown"):
            reason = _cuda_launch.version_gate(version)
            assert reason is not None, version
            assert "outside" in reason, version
        # on skips the gate (and builds), off never builds, other values raise.
        triton = types.SimpleNamespace(__version__="3.10.0")
        build = mock.patch.object(_extensions, "load", return_value="launcher")
        with build, mock.patch.dict(sys.modules, {"triton": triton}):
            for value, expected in (("on", ("launcher", "")), ("1", ("launcher", ""))):
                with _setting(value):
                    assert _cuda_launch.status() == expected
            for value in ("off", "0"):
                with _setting(value):
                    assert _cuda_launch.status() == (None, "SDCA_CUDA_LAUNCHER=off")
            with _setting("auto"):
                assert "outside" in _cuda_launch.status()[1]
            with _setting("yes"), pytest.raises(ValueError, match="auto, on or off"):
                _cuda_launch.status()
        assert _cuda_launch.kernels_supported("3.3.0")
        assert _cuda_launch.kernels_supported("3.10.0")
        assert not _cuda_launch.kernels_supported("3.2.0")

    def test_every_fused_kernel_needs_triton_3_3(self):
        """Each fused-kernel entry declines Triton 3.2, which aborts compiling them."""
        triton = pytest.importorskip("triton")
        from libsdca import _cross_entropy, _entropy, _hinge, _multilabel, _topk
        from libsdca import torch as sdca_torch

        gates = (
            _entropy._kernel,
            _cross_entropy._weighted_kernel,
            _hinge._kernel,
            _topk._kernel,
            _multilabel._kernel,
            sdca_torch._kernel_prox,
            sdca_torch._kernel_distill,
        )

        def kernels() -> list[bool]:
            """Whether each entry returns a kernel."""
            for gate in gates:
                gate.cache_clear()
            found = [gate() for gate in gates] + [sdca_torch._kernel_distill(True)]
            return [kernel is not None for kernel in found]

        try:
            with mock.patch.object(torch.version, "cuda", "12.6"):
                with mock.patch.object(triton, "__version__", "3.2.0"):
                    assert not any(kernels())
                with mock.patch.object(triton, "__version__", "3.3.0"):
                    assert all(kernels())
        finally:
            for gate in gates:
                gate.cache_clear()

    def test_kernel_gate(self):
        plain = types.SimpleNamespace(name="k", num_ctas=1, cluster_dims=(1, 1, 1))
        own = (8, 8, 8, 4)
        for params in (None, [8, 8, 8, 4], [8, 8, 8, 4, 8], [8, 8, 8, 4, 8, 8]):
            assert _cuda_launch.kernel_gate(plain, params, own) is None, params
        for params in ([8, 8, 4], [8, 8, 8, 8], [8, 8, 8, 4, 8, 8, 8], [8, 8, 8, 4, 4]):
            reason = _cuda_launch.kernel_gate(plain, params, own)
            assert reason is not None, params
            assert "parameters" in reason, params
        for change in (
            {"num_ctas": 2},
            {"cluster_dims": (2, 1, 1)},
            {"global_scratch_size": 128},
            {"profile_scratch_size": 64},
            {"launch_cooperative_grid": True},
            {"launch_pdl": True},
            {"instrumentation_mode": "gsan"},
        ):
            metadata = types.SimpleNamespace(**dict(vars(plain), **change))
            assert _cuda_launch.kernel_gate(metadata, None, own) is not None, change

    def test_build_failure_is_recorded_and_expires(self, tmp_path: Path):
        triton = types.SimpleNamespace(__version__="3.5.0")
        failing = mock.patch("torch.utils.cpp_extension.load", side_effect=RuntimeError("no nvcc"))
        tools = mock.patch.object(_extensions, "missing_tools", return_value=None)
        modules = mock.patch.dict(sys.modules, {"triton": triton})
        directory = mock.patch.dict(os.environ, {"TORCH_EXTENSIONS_DIR": str(tmp_path)})

        def resolve() -> tuple[tuple[object, str], int]:
            _cuda_launch.status.cache_clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                first = _cuda_launch.status()
                assert _cuda_launch.status() is first  # once per process
            return first, len([w for w in caught if w.category is RuntimeWarning])

        with failing, tools, modules, directory, _setting("auto"):
            (host, reason), warned = resolve()
            assert host is None and warned == 1
            assert "did not build (RuntimeError: no nvcc)" in reason
            record = _cuda_launch.failure_record("3.5.0")
            assert record.is_file() and str(record) in reason
            # A later process skips the build without warning; explain_backend's reason
            # names the record.
            (host, reason), warned = resolve()
            assert host is None and warned == 0
            assert reason.startswith("an earlier build failed") and str(record) in reason
            # After RETRY seconds the record expires and the build runs again.
            old = time.time() - _extensions.RETRY - 60
            os.utime(record, (old, old))
            (host, reason), warned = resolve()
            assert warned == 1 and "did not build" in reason
            # on raises instead.
            with _setting("on"), pytest.raises(RuntimeError, match="no nvcc"):
                _cuda_launch.status()

    def test_missing_tools_hint_once(self):
        triton = types.SimpleNamespace(__version__="3.5.0")
        tools = mock.patch.object(_extensions, "missing_tools", return_value="no CUDA headers")
        build = mock.patch.object(_extensions, "_build")
        with tools, build as built, mock.patch.dict(sys.modules, {"triton": triton}):
            with _setting("auto"), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert _cuda_launch.status() == (None, "no CUDA headers")
                _cuda_launch.status()
            built.assert_not_called()
        hints = [str(w.message) for w in caught if "SDCA_CUDA_LAUNCHER=off" in str(w.message)]
        assert len(hints) == 1
        assert "launch through Triton: no CUDA headers" in hints[0]

    def test_on_raises_where_tools_are_missing(self):
        """on names the missing tool instead of falling back; the version gate stays skipped."""
        triton = types.SimpleNamespace(__version__="3.10.0")
        tools = mock.patch.object(_extensions, "missing_tools", return_value="no CUDA headers")
        build = mock.patch.object(_extensions, "_build")
        with tools, build as built, mock.patch.dict(sys.modules, {"triton": triton}):
            for value in ("on", "1"):
                with _setting(value), pytest.raises(RuntimeError) as raised:
                    _cuda_launch.status()
                assert "the C++ launcher cannot be built here: no CUDA headers" in str(raised.value)
            built.assert_not_called()

    def test_tools_check(self, tmp_path: Path):
        """The launcher's tools: the CPU extension's, then cuda.h under CUDA_HOME and
        libcuda.so.1."""
        common = mock.patch("shutil.which", return_value="/usr/bin/tool")
        python_h = tmp_path / "Python.h"
        python_h.touch()
        paths = mock.patch("sysconfig.get_paths", return_value={"include": str(tmp_path)})
        home = mock.patch.object(_extensions, "cuda_home", return_value=tmp_path / "cuda")
        loaded = mock.patch("ctypes.CDLL")
        with common, paths, home, loaded as cdll, mock.patch.object(sys, "platform", "linux"):
            assert _extensions.missing_tools() is None
            reason = _extensions.missing_tools(cuda=True)
            assert reason is not None and "CUDA headers" in reason and "CUDA_HOME" in reason
            (tmp_path / "cuda/include").mkdir(parents=True)
            (tmp_path / "cuda/include/cuda.h").write_text("#define CUDA_VERSION 12060\n")
            assert _extensions.missing_tools(cuda=True) is None
            cdll.assert_called_with("libcuda.so.1")
            assert _extensions.cuda_version() == "12060"
            cdll.side_effect = OSError("libcuda.so.1: cannot open shared object file")
            reason = _extensions.missing_tools(cuda=True)
            assert reason is not None and "libcuda.so.1" in reason
            python_h.unlink()
            reason = _extensions.missing_tools(cuda=True)
            assert reason is not None and "Python.h" in reason

    def test_build_directory_names_everything_that_changes_the_binary(self, tmp_path: Path):
        """The cache key: source, PyTorch, Python, compiler and its version, flags, CUDA and
        Triton versions; the failure record sits next to the build directory."""
        with mock.patch.dict(os.environ, {"TORCH_EXTENSIONS_DIR": str(tmp_path)}):
            base = _cuda_launch._extension("3.5.0")
            directory = _extensions.build_directory(base)
            assert directory.parent == tmp_path
            assert directory.name.startswith("libsdca_entropy_cuda_")
            assert _cuda_launch.failure_record("3.5.0") == tmp_path / (
                directory.name + ".build-failed.txt"
            )
            assert _extensions.build_directory(base) == directory
            changes = [
                _cuda_launch._extension("3.6.0"),
                dataclasses.replace(base, key=(base.key[0], "cuda 12080 12.8")),
                dataclasses.replace(base, cflags=(*base.cflags, "-g")),
                dataclasses.replace(base, ldflags=base.ldflags[:-1]),
            ]
            names = {_extensions.build_directory(change).name for change in changes}
            assert directory.name not in names and len(names) == len(changes)
            with mock.patch.object(_extensions, "_compiler", return_value="c++\nother 1.0"):
                assert _extensions.build_directory(base) != directory
            with mock.patch.object(torch, "__version__", "0.0.0"):
                assert _extensions.build_directory(base) != directory
            with mock.patch.object(sys.implementation, "cache_tag", "cpython-0"):
                assert _extensions.build_directory(base) != directory
            source = mock.patch.object(Path, "read_bytes", return_value=b"other source")
            with source:
                assert _extensions.build_directory(base) != directory

    def test_build_renames_a_private_directory_into_place(self, tmp_path: Path):
        """A build goes to a private sibling directory renamed into place when it is
        complete; where another process renamed its build first, that build stays."""
        extension = dataclasses.replace(_cuda_launch._extension("3.5.0"), cuda=False)
        names = []

        def build(name: str, build_directory: str, **_: object) -> str:
            assert Path(build_directory).name.startswith(name + ".")
            (Path(build_directory) / f"{name}.so").write_text("library")
            names.append(name)
            return "module"

        compile_ = mock.patch("torch.utils.cpp_extension.load", side_effect=build)
        tools = mock.patch.object(_extensions, "missing_tools", return_value=None)
        directory_root = mock.patch.dict(os.environ, {"TORCH_EXTENSIONS_DIR": str(tmp_path)})
        with compile_, tools, directory_root, mock.patch.dict(_extensions._modules, clear=True):
            directory = _extensions.build_directory(extension)
            assert _extensions.load(extension) == "module"
            assert _extensions.load(extension) == "module"  # once per process
            assert names == [directory.name]
            assert (directory / f"{directory.name}.so").read_text() == "library"
            assert [p.name for p in tmp_path.iterdir()] == [directory.name]
            # Another process got there first: its build stays, the private one goes.
            _extensions._modules.clear()
            (directory / f"{directory.name}.so").write_text("first")
            assert _extensions._build(extension, directory) == "module"
            assert (directory / f"{directory.name}.so").read_text() == "first"
            assert [p.name for p in tmp_path.iterdir()] == [directory.name]

    def test_auto_falls_back_in_the_interpreter(self):
        """auto without the tools launches the fused kernels through Triton (one hint, the
        results of off); on raises naming the tool. TRITON_INTERPRET=1 on CPU tensors."""
        stdout = interpreter.run(
            """
            import os, warnings
            from unittest import mock
            import torch
            from libsdca import _cuda_launch, _extensions, _triton_entropy

            z = 3 * torch.randn(8, 37, dtype=torch.float64, generator=torch.manual_seed(0))
            y = torch.randint(0, 37, (8,))
            missing = mock.patch.object(_extensions, "missing_tools", return_value="no tools")
            gate = mock.patch.object(_cuda_launch, "version_gate", return_value=None)
            values = {}
            with missing, gate, warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for mode in ("off", "auto", "on"):
                    os.environ[_cuda_launch.ENV] = mode
                    _cuda_launch.status.cache_clear()
                    try:
                        values[mode] = [_triton_entropy.forward(z, y, "none")[0] for _ in "ab"]
                    except RuntimeError as error:
                        print("RAISED", error)
            assert torch.equal(values["off"][0], values["auto"][1])
            print("HINTS", sum("launch through Triton: no tools" in str(w.message) for w in caught))
            """,
            timeout=600,
        )
        assert "RAISED libsdca: the C++ launcher cannot be built here: no tools" in stdout
        assert "HINTS 1" in stdout

    def test_reduction_in_the_interpreter(self):
        """The reduction's control flow on CPU with small blocks, numpy in place of libdevice."""
        interpreter.run(
            """
            import test_cuda_launch as t
            from libsdca import _triton_reduce as reduce

            # One-launch up to 64 rows in chunks of 8; partials of 16 rows (two finish passes
            # over partials above 256 rows).
            reduce.BLOCK, reduce.DIRECT_ROWS, reduce.DIRECT_BLOCK = 16, 64, 8
            t.check_reduction(reduce, "cpu", [1, 2, 9, 64, 65, 300])
            """
        )

    def test_reduction_reference(self):
        assert reduction_reference([MAX / 2] * 4, [0.0] * 4, "mean", True) == MAX / 2
        assert reduction_reference([MAX / 2] * 4, [0.0] * 4, "sum", False) == math.inf
        value = reduction_reference(
            [math.inf, 0.0], [math.log(MAX) + math.log(2), -1.0], "mean", True
        )
        assert round(abs(value / MAX - 1.0), 12) == 0
        tiny = reduction_reference([1e-310] * 3, [math.log(1e-310)] * 3, "sum", True)
        assert round(abs(tiny / 3e-310 - 1.0), 9) == 0

    def test_int32_offsets_fit_rows(self):
        """Each k * N of the Triton kernels at ROWS rows: below 2**31 unless N is int64."""
        spec = importlib.util.find_spec("libsdca")
        assert spec is not None and spec.origin is not None
        root = Path(spec.origin).parent
        trees = [ast.parse(path.read_text()) for path in sorted(root.glob("_triton_*.py"))]
        nodes = [n for tree in trees for n in ast.walk(tree)]
        layouts = []
        for node in nodes:
            if isinstance(node, ast.Assign) and ast.unparse(node).startswith("_LAYOUT"):
                value = ast.literal_eval(node.value)
                layouts += value.values() if isinstance(value, dict) else [value]
        # The reduction's (VALUE, LOG, BAD, STATUS) slots: the largest of any layout.
        names = ("VALUE", "LOG", "BAD", "STATUS")
        slots = {name: max(layout[i] for layout in layouts) for i, name in enumerate(names)}
        assert slots["STATUS"] == 13
        counts = [0, 0]
        for function in (n for n in nodes if isinstance(n, ast.FunctionDef)):
            wide = any(ast.unparse(s) == "N = N.to(tl.int64)" for s in function.body)
            for node in ast.walk(function):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                    continue
                left, right = ast.unparse(node.left), ast.unparse(node.right)
                factor = right if left == "N" else left if right == "N" else None
                if factor is None or "." in factor:  # not N, or a float scale
                    continue
                counts[wide] += 1
                if not wide:
                    factor = slots[factor] if factor in slots else int(factor)
                    assert factor * _cuda_launch.ROWS < 2**31, function.name
        assert counts[0] > 40
        assert counts[1] > 5

    def test_handles_stay_on_the_cpu(self):
        # The C++ launcher reads the handles on the host, whatever the default device.
        with torch.device("meta"):
            handles = _cuda_launch.handles([], 7, 8)
        assert handles.device.type == "cpu"
        assert handles.tolist() == [7, 8]

    def test_explain_backend_on_cpu(self):
        z = torch.randn(6, 5, dtype=torch.float32)
        y = torch.randint(0, 5, (6,))
        mask = F.one_hot(y, 5).bool()
        kernel = _cpu_entropy.enabled()
        assert explain_backend(z, y)[0] == ("cpu-kernel" if kernel else "native")
        # float64 integer entropy: the extension's rows of at most 32 classes, the native
        # library's wider ones (the same statistics and double extremes either way).
        assert explain_backend(z.double(), y)[0] == ("cpu-kernel" if kernel else "native")
        wide = torch.randn(6, 33, dtype=torch.float64)
        assert explain_backend(wide, y)[0] == "native"
        assert explain_backend(z, y, backend="torch")[0] == "torch"
        assert explain_backend(z, y, backend="native")[0] == "native"
        assert explain_backend(z, mask)[0] == ("cpu-kernel" if kernel else "native")
        assert explain_backend(z, mask, backend="torch")[0] == "torch"
        hinge = explain_backend(z, y, "topk_hinge_alpha")[0]
        assert hinge == ("cpu-kernel" if kernel else "native")
        name, reason = explain_backend(z, y, "topk_hinge_beta", k=2, backend="torch")
        assert name == "torch"
        assert "backend='torch'" in reason
        if not kernel:
            assert "the CPU extension is unavailable" in explain_backend(z, y)[1]
        with pytest.raises(ValueError):
            explain_backend(z, y, "unknown")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _triton()), reason="CUDA with Triton is not available"
)
class FusedLaunchTests:
    def setup_method(self):
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)

    def teardown_method(self):
        _cuda_launch.status.cache_clear()

    def _data(
        self, n: int, c: int, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = 3 * torch.randn(n, c, device=self.device, dtype=dtype, generator=self.gen)
        y = torch.randint(0, c, (n,), device=self.device, generator=self.gen)
        return z, y

    def _counts(self):
        _, entropy, hinge, reduce = _loaded_triton()
        kernels = [
            entropy._forward,
            entropy._backward,
            hinge._forward,
            hinge._backward,
            reduce.partials,
            reduce.finish,
        ]
        compiled = (entropy._compiled.cache_info().currsize, hinge._compiled.cache_info().currsize)
        return compiled, [_variants(k) for k in kernels]

    def test_variable_batch_sizes_reuse_kernels(self):
        sizes = [1, 2, 3, 15, 16, 17, 31, 255, 1000, 2047, 2048, 2049, 4097, 10001, 40000]
        for setting in ("auto", "off"):
            with _setting(setting):
                # One-launch and multi-block reductions compile separate kernels.
                for n in (7, 20000):
                    for loss in ("cross_entropy", *HINGES):
                        for reduction in ("none", "mean", "sum"):
                            _fused(*self._data(n, 33), loss, reduction)
                before = self._counts()
                for n in sizes:
                    z, y = self._data(n, 33)
                    for loss in ("cross_entropy", *HINGES):
                        for reduction in ("none", "mean", "sum"):
                            value, grad = _fused(z, y, loss, reduction)
                            ref_value, ref_grad = _reference(z, y, loss, reduction)
                            label = f"{setting} {loss} n={n} {reduction}"

                            def message(detail: str, label: str = label) -> str:
                                return f"{label}: {detail}"

                            # The smooth projection is accurate to the margins' scale: a
                            # tiny a keeps an absolute, not a relative, rounding error.
                            atol = 1e-15 if loss != "cross_entropy" else 1e-300
                            torch.testing.assert_close(
                                value, ref_value, rtol=1e-12, atol=atol, msg=message
                            )
                            torch.testing.assert_close(
                                grad, ref_grad, rtol=1e-12, atol=atol, msg=message
                            )
                assert self._counts() == before, setting

    def test_compiled_handles_are_bounded(self):
        _, entropy, _, _ = _loaded_triton()
        assert entropy._compiled.cache_info().maxsize == _cuda_launch.CACHE

    def test_large_batch(self):
        # More rows than one reduction block of blocks: the finish loop runs twice.
        n = 2**22 + 3
        z, y = self._data(n, 3, torch.float32)
        z[n // 3] = torch.tensor([0.0, 1e30, -1e30])
        for loss in ("cross_entropy", *HINGES):
            name, reason = explain_backend(z, y, _case(loss)[0], gamma=_case(loss)[1])
            assert name in ("cuda-launcher", "triton"), reason
            for reduction in ("mean", "sum", "none"):
                value, grad = _fused(z, y, loss, reduction)
                ref_value, ref_grad = _reference(z, y, loss, reduction)
                message = f"{loss} {reduction}"
                rtol = 2e-6 if loss == "cross_entropy" else 2e-7
                torch.testing.assert_close(
                    value.double(), ref_value, rtol=rtol, atol=0, msg=message
                )
                torch.testing.assert_close(
                    grad.double(), ref_grad, rtol=rtol, atol=1e-30, msg=message
                )
                again, _ = _fused(z, y, loss, reduction)
                assert torch.equal(value, again), message

    def test_large_batch_reduction_boundaries(self):
        n, maximum = 2**21 + 1, torch.finfo(torch.float64).max
        y = torch.zeros(n, dtype=torch.long, device=self.device)
        # Sums overflow while the mean is finite.
        row = torch.tensor([[0.0, maximum / 4]], dtype=torch.float64, device=self.device)
        z = row.expand(n, 2).contiguous()
        for loss in ("cross_entropy", "topk_hinge_alpha"):
            value = sdca_loss(z, y, loss)
            assert round(abs(value.item() / (maximum / 4) - 1.0), 12) == 0
            assert sdca_loss(z, y, loss, reduction="sum").item() == float("inf")
        # Losses below the normal range (about 1.2e-308) sum in the log domain.
        row = torch.tensor([[0.0, -709.0]], dtype=torch.float64, device=self.device)
        z = row.expand(n, 2).contiguous()
        single = sdca_loss(z[:1], y[:1], reduction="sum").item()
        assert 0 < single
        assert single < torch.finfo(torch.float64).tiny
        assert round(abs(sdca_loss(z, y).item() / single - 1.0), 12) == 0
        assert round(abs(sdca_loss(z, y, reduction="sum").item() / (n * single) - 1.0), 12) == 0
        # One loss beyond double's range among ordinary rows: a finite log-domain mean.
        z = torch.zeros(n, 2, dtype=torch.float64, device=self.device)
        z[n - 2] = torch.tensor([-maximum, maximum], dtype=torch.float64)
        value = sdca_loss(z, y).item()
        # 2 * maximum overflows a Python float; maximum / n does not.
        assert round(abs(value / (maximum / n * 2) - 1.0), 9) == 0
        # Invalid rows in the last block propagate NaN without host synchronization.
        z = torch.zeros(n, 2, dtype=torch.float32, device=self.device)
        z[n - 1, 1] = float("nan")
        for loss in ("cross_entropy", "topk_hinge_alpha"):
            assert bool(sdca_loss(z, y, loss).isnan().all())

    def test_reduction_kernels_alone(self):
        # One launch up to DIRECT_ROWS rows, block partials above.
        _, _, _, reduce = _loaded_triton()
        rows = reduce.DIRECT_ROWS
        check_reduction(reduce, "cuda", [1, 7, rows, rows + 1, 40000])

    def test_forced_triton_launch_matches(self):
        for n in (5, 3000):
            z, y = self._data(n, 1000, torch.float32)
            for loss in ("cross_entropy", *HINGES):
                for reduction in ("none", "mean"):
                    default = _fused(z, y, loss, reduction)
                    with _setting("off"):
                        name, gamma = _case(loss)
                        name, reason = explain_backend(z, y, name, gamma=gamma, reduction=reduction)
                        assert name == "triton"
                        assert "SDCA_CUDA_LAUNCHER=off" in reason
                        forced = _fused(z, y, loss, reduction)
                    message = f"{loss} n={n} {reduction}"
                    assert torch.equal(default[0], forced[0]), message
                    assert torch.equal(default[1], forced[1]), message

    def test_version_gate_falls_back(self):
        triton = _loaded_triton()[0]
        z, y = self._data(9, 10)
        expected = _fused(z, y, "cross_entropy", "mean")
        # The version gate applies only when SDCA_CUDA_LAUNCHER does not force a path.
        with mock.patch.dict(os.environ), mock.patch.object(triton, "__version__", "3.99.0"):
            os.environ.pop(_cuda_launch.ENV, None)
            _cuda_launch.status.cache_clear()
            name, reason = explain_backend(z, y)
            assert name == "triton"
            assert "outside" in reason
            actual = _fused(z, y, "cross_entropy", "mean")
        _cuda_launch.status.cache_clear()
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])

    def test_explain_backend_on_cuda(self):
        z, y = self._data(4, 6)
        host, reason = _cuda_launch.status()
        name = explain_backend(z, y)[0]
        assert name == ("cuda-launcher" if host is not None else "triton"), reason
        mask = F.one_hot(y, 6).bool()
        assert explain_backend(z, mask)[0] == name
        assert explain_backend(z.float(), mask, precision="float32")[0] == "torch"
        assert explain_backend(z, y, "topk_hinge_beta", k=2)[0] == name
        # precision="float32" takes non-float64 logits, as in sdca_loss.
        single = z.float()
        assert explain_backend(single, y, precision="float32")[0] == name
        assert explain_backend(single, y, "topk_hinge_alpha", precision="float32")[0] == "torch"
        with pytest.raises(ValueError, match="precision"):
            explain_backend(z, y, precision="float32")
        assert explain_backend(z, y, "proximal_cross_entropy")[0] == "triton"

    def test_cuda_default_device(self):
        # A fresh process, so each family builds its kernel handles under the CUDA default.
        script = textwrap.dedent(
            """
            import torch
            import torch.nn.functional as F
            from libsdca import _cuda_launch
            from libsdca.torch import cross_entropy, explain_backend, sdca_loss

            host, reason = _cuda_launch.status()
            if host is None:
                print("SKIP", reason)
                raise SystemExit
            gen = torch.Generator(device="cuda").manual_seed(0)
            z = 3 * torch.randn(64, 50, device="cuda", generator=gen)
            y = torch.randint(0, 50, (64,), device="cuda", generator=gen)
            mask = F.one_hot(y, 50).bool()
            mask[:, 0] = True
            w = torch.rand(50, device="cuda", generator=gen) + 0.5
            calls = {
                "cross_entropy": lambda x: sdca_loss(x, y),
                "weighted": lambda x: cross_entropy(x, y, weight=w),
                "hinge": lambda x: sdca_loss(x, y, "topk_hinge_alpha"),
                "topk": lambda x: sdca_loss(x, y, "topk_hinge_beta", k=2),
                "multilabel": lambda x: sdca_loss(x, mask, "multilabel_hinge"),
            }

            def run():
                out = {}
                for name, call in calls.items():
                    x = z.detach().clone().requires_grad_()
                    value = call(x)
                    value.backward()
                    out[name] = (value.detach(), x.grad)
                return out

            for target, loss, k in ((y, "cross_entropy", 1), (y, "topk_hinge_alpha", 1),
                                    (y, "topk_hinge_beta", 2), (mask, "multilabel_hinge", 1)):
                assert explain_backend(z, target, loss, k=k)[0] == "cuda-launcher", loss
            torch.set_default_device("cuda")
            first = run()
            torch.set_default_device("cpu")
            second = run()
            for name in calls:
                assert torch.equal(first[name][0], second[name][0]), name
                assert torch.equal(first[name][1], second[name][1]), name
            print("OK")
            """
        )
        env = dict(os.environ)
        env.pop(_cuda_launch.ENV, None)
        result = run_python("-W", "ignore", "-c", script, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-3000:]
        if result.stdout.startswith("SKIP"):
            pytest.skip(result.stdout.strip())
        assert "OK" in result.stdout

    def test_inference_mode_gamma_cache_reused_in_training(self):
        z, y = self._data(4, 7)
        mask = F.one_hot(y, 7).bool()
        mask[:, 0] = True
        mask[:, 1] = False
        for setting in ("auto", "off"):
            with _setting(setting):
                for loss, target, k, gamma in (
                    ("cross_entropy", y, 1, 1.0),
                    ("topk_hinge_beta", y, 1, 0.37),
                    ("topk_hinge_beta", y, 2, 0.39),
                    ("multilabel_hinge", mask, 1, 0.41),
                ):
                    with torch.inference_mode():
                        sdca_loss(z, target, loss, k=k, gamma=gamma)
                    x = z.detach().clone().requires_grad_()
                    value = sdca_loss(x, target, loss, k=k, gamma=gamma)
                    value.backward()
                    assert x.grad is not None
                    assert bool(x.grad.isfinite().all())

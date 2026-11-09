"""Smooth hinge projections terminate on near-tie rows on every torch backend.

Each backend runs in a watched subprocess: a projection loop that cycles fails the test
instead of hanging it (a cycling CUDA kernel blocks the process until it is killed).
"""

import importlib
import importlib.util
import os
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest
import torch
from child_process import pythonpath
from libsdca import loss_and_grad

# A float64 row on which Michelot's set alternated between 6 and 5 margins: removing the
# margin at the threshold rounds the threshold one ulp below it, which re-admitted it.
ROW = [
    -0.48925711930947796,
    0.0,
    -0.24067951835045504,
    -0.0189320465663128,
    -0.035375611101318316,
    -0.22828873347377265,
    0.0,
]
GAMMA = 1.9230096870555307
CASES = (
    ("topk_hinge_alpha", 1),
    ("topk_hinge_beta", 1),
    ("topk_hinge_alpha", 2),
    ("topk_hinge_alpha", 5),
    ("topk_hinge_beta", 2),
    ("topk_hinge_beta", 5),
)
# Seconds to import and compile, then to run every case on the crafted rows.
COMPILE, RUN = 1200, 120


def crafted(loss: str, k: int, n: int, c: int, gamma: float, seed: int):
    """n rows with target c - 1 and one margin at the projection threshold of the others."""
    rng = np.random.default_rng(seed)
    z = -rng.uniform(0, 1, (n, c)) * rng.choice([0.3, 1.0, 3.0], (n, 1))
    z[:, -1] = 0.0
    y = np.full(n, c - 1)
    rows, e = np.arange(n), rng.integers(0, c - 1, n)
    z[rows, e] = -1e3
    # a = gamma * gradient; a free margin j (0 < a_j < cap) has threshold h_j - a_j.
    a = loss_and_grad(z, y, loss, k=k, gamma=gamma)[1][:, :-1] * gamma
    free = (a > 0) & ((a < a.max(1, keepdims=True)) | (k == 1))
    j = np.where(free, a, -1.0).argmax(1)
    tau = (1 + z[rows, j]) - a[rows, j]
    z[rows, e] = np.where(free.any(1), tau - 1, z[rows, e])
    return z, y


def batches(loss: str, k: int, interpret: bool):
    """A row that once failed to terminate (k = 1) and crafted rows; gamma = 8 puts
    hinge_topk_smooth in the cone."""
    out = [(np.array([ROW]), np.array([6]), GAMMA)] if k == 1 else []
    n = 8 if interpret else 1000
    for seed, (c, gamma) in enumerate(((8, 0.3), (24, GAMMA), (16, 8.0))):
        if c > k + 1:
            out.append((*crafted(loss, k, n, c, gamma, seed), gamma))
    return out


def _evaluate(backend: str, z: np.ndarray, y: np.ndarray, loss: str, k: int, gamma: float):
    """Per-row values and gradients (float64 numpy) of one backend."""
    if backend == "interpret":
        import interpreter  # noqa: F401  # the libdevice shims of the interpreter
        from libsdca import _triton_hinge, _triton_topk
        from libsdca._core import _kind

        zt, yt = torch.from_numpy(z), torch.from_numpy(y)
        ones = torch.ones(len(zt), dtype=zt.dtype)
        if k == 1:
            value, stats = _triton_hinge.forward(zt, yt, "none", True, gamma)
            gradient = _triton_hinge.backward(zt, yt, stats, ones, "none", True, gamma)
        else:
            # The kernels take the private kind: the public name with gamma > 0 is smooth.
            kind = _kind(loss, gamma)
            value, stats = _triton_topk.forward(zt, yt, "none", kind, k, gamma)
            gradient = _triton_topk.backward(zt, yt, stats, ones, "none", kind, k, gamma)
        return value.numpy(), gradient.numpy()
    from libsdca.torch import explain_backend, sdca_loss

    device = "cpu" if backend in ("native", "torch", "cpu-kernel") else "cuda"
    option = "torch" if backend in ("torch", "cuda-torch") else "auto"
    x = torch.tensor(z, device=device, requires_grad=True)
    t = torch.tensor(y, device=device)
    if backend == "cuda-torch":
        # A one-hot mask keeps CUDA off the fused kernels, which take class indices.
        t = torch.nn.functional.one_hot(t, z.shape[1]).bool()
    taken = explain_backend(x, t, loss, k=k, reduction="none", backend=option)[0]
    if taken != ("torch" if backend == "cuda-torch" else backend):
        print(f"skip {backend}: {loss} k={k} takes {taken}", flush=True)
        sys.exit(0)
    value = sdca_loss(x, t, loss, k=k, gamma=gamma, reduction="none", backend=option)
    value.backward(torch.ones_like(value))
    assert x.grad is not None
    return value.detach().cpu().numpy(), x.grad.cpu().numpy()


def _check(
    value: float | np.ndarray,
    gradient: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    loss: str,
    k: int,
    gamma: float,
):
    """Within 1e-12 of loss.h: values (or their sum; margins are O(1), so an absolute
    floor of 1e-15), and gradients relative to their row's largest."""
    ref_value, ref_gradient = loss_and_grad(z, y, loss, k=k, gamma=gamma)
    ref_value = ref_value.sum() if np.ndim(value) == 0 else ref_value
    np.testing.assert_allclose(value, ref_value, rtol=1e-12, atol=1e-15)
    tol = 1e-12 * np.abs(ref_gradient).max(1, keepdims=True)
    rows = np.nonzero((np.abs(gradient - ref_gradient) > tol).any(1))[0]
    assert rows.size == 0, f"{loss} k={k} gamma={gamma}: gradients differ in rows {rows[:8]}"


def _child(backend: str):
    """Compile on benign rows, print warm, then check the near-tie rows against loss.h."""
    interpret = backend == "interpret"
    work = {case: batches(*case, interpret) for case in CASES}
    for (loss, k), items in work.items():
        for z, y, gamma in items:
            _evaluate(backend, np.zeros((2, z.shape[1])), np.full(2, y[0]), loss, k, gamma)
    print("warm", flush=True)
    for (loss, k), items in work.items():
        for z, y, gamma in items:
            _check(*_evaluate(backend, z, y, loss, k, gamma), z, y, loss, k, gamma)
    print("done", flush=True)


def _jax_child():
    """The JAX tensor losses (backend="jax", float64) on the same rows."""
    # By name: an import statement would put this file in the jax test group.
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    jax_loss = importlib.import_module("libsdca.jax").sdca_loss

    jax.config.update("jax_enable_x64", True)
    print("warm", flush=True)
    for loss, k in CASES:
        for z, y, gamma in batches(loss, k, False):

            def total(
                x: object, y: np.ndarray = y, loss: str = loss, k: int = k, gamma: float = gamma
            ):
                return jax_loss(x, y, loss, k=k, gamma=gamma, reduction="sum", backend="jax")

            value, gradient = jax.value_and_grad(total)(jnp.asarray(z))
            _check(float(value), np.asarray(gradient), z, y, loss, k, gamma)
    print("done", flush=True)


def watch(backend: str, env: dict[str, str]):
    """Run _child(backend); fail if a phase overruns its time, or the child fails."""
    call = "_jax_child()" if backend == "jax" else f"_child({backend!r})"
    code = f"import test_projection_termination as t; t.{call}"
    env = dict(os.environ, **env)
    env["PYTHONPATH"] = pythonpath(env)
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=tempfile.gettempdir(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout = proc.stdout
    assert stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()
    log: list[str] = []

    def read():
        for line in stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    try:
        for marker, limit in (("warm", COMPILE), ("done", RUN)):
            while True:
                try:
                    line = lines.get(timeout=limit)
                except queue.Empty:
                    pytest.fail(f"{backend}: no '{marker}' within {limit} s:\n{''.join(log)}")
                if line is None:
                    proc.wait()
                    if any(entry.startswith("skip ") for entry in log):
                        pytest.skip("".join(log).strip())
                    pytest.fail(f"{backend}: exit {proc.returncode}:\n{''.join(log)}")
                log.append(line)
                if line.strip() == marker:
                    break
        assert proc.wait(timeout=60) == 0, "".join(log)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stdout.close()


def _cuda_triton():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _interpretable():
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return not torch.cuda.is_available()


class ProjectionTerminationTests:
    def test_native(self):
        watch("native", {})

    def test_torch_tensor(self):
        watch("torch", {})

    def test_cpu_kernel(self):
        # Builds the CPU extension; skipped where it does not build.
        from libsdca import _cpu_entropy

        if not _cpu_entropy.buildable():
            pytest.skip("the CPU extension does not build here")
        watch("cpu-kernel", {"SDCA_CPU_EXTENSION": "on"})

    @pytest.mark.slow
    @pytest.mark.skipif(not _interpretable(), reason="needs Triton on a machine without CUDA")
    def test_triton_interpreter(self):
        watch("interpret", {"TRITON_INTERPRET": "1"})

    @pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA with Triton")
    def test_cuda_triton(self):
        watch("triton", {"SDCA_CUDA_LAUNCHER": "off"})

    @pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA with Triton")
    def test_cuda_launcher(self):
        watch("cuda-launcher", {"SDCA_CUDA_LAUNCHER": "on"})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_tensor(self):
        watch("cuda-torch", {})

    @pytest.mark.slow
    @pytest.mark.skipif(not importlib.util.find_spec("jax"), reason="needs JAX")
    def test_jax(self):
        watch("jax", {})

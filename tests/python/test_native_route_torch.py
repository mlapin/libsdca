"""The native route of libsdca.torch: without its CPU extension (SDCA_CPU_EXTENSION=off, or
no compiler) CPU calls run the same kernels in the native library (sdca_loss_rows), as the
CPU kernels of the libsdca::loss operators. A child without the extension gives the
extension's bytes for every loss, option and dtype of native_route_cases, except where a
float32 or float64 upstream derivative is not 1: there the native route's gradient,
saved by the forward at upstream 1, is scaled and rounded again (as in compiled graphs). A
missing compiler warns once, and explain_backend names the native library and why.

The cross-interface and rule-table tests run on the native route too:
SDCA_CPU_EXTENSION=off SDCA_JAX_FFI=off pytest tests/python/test_cross_interface.py
tests/python/test_rule_table.py -k "torch or jax".
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import libsdca.torch as L
import native_route_cases as cases
import pytest
import torch
from child_process import run_python
from libsdca import _torch_ops
from libsdca._frontend import KINDS, Rows

CPU = torch.device("cpu")
NO_COMPILER = "the CPU kernels of libsdca.torch cannot be built here: no C++ compiler"


def child(directory: Path, outputs: str = "outputs", **env: str) -> dict[str, Any]:
    path = directory / "torch.pkl"
    result = run_python(
        "-m", "native_route_cases", "torch", str(path), outputs,
        env={**os.environ, **env}, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-4000:]
    return pickle.loads(path.read_bytes())


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return child(tmp_path_factory.mktemp("native"), SDCA_CPU_EXTENSION="off")


@pytest.fixture(scope="module")
def extension() -> dict[str, tuple[bytes, bytes]]:
    reason = _torch_ops.cpu_kernels() or _torch_ops.fallback(CPU)
    if reason is not None:
        pytest.skip(f"no CPU extension to compare with: {reason}")
    return cases.torch_outputs()


def scaled_case(key: str) -> bool:
    return key.endswith(f"-{cases.SCALE}")


def test_the_native_route_gives_the_extensions_bytes(
    native: dict[str, Any], extension: dict[str, tuple[bytes, bytes]]
) -> None:
    outputs = native["outputs"]
    assert outputs.keys() == extension.keys()
    differ = [
        key
        for key, want in extension.items()
        if outputs[key] != want
        and (not scaled_case(key) or getattr(torch, key.split("-")[1]) in _torch_ops.HALF)
    ]
    assert not differ


def _gradient(raw: bytes, dtype: torch.dtype) -> torch.Tensor:
    stored = torch.int16 if dtype.itemsize == 2 else dtype
    return torch.frombuffer(bytearray(raw), dtype=stored).view(dtype).reshape(cases.N, cases.C)


def test_other_upstream_derivatives_scale_the_saved_gradient(
    native: dict[str, Any], extension: dict[str, tuple[bytes, bytes]]
) -> None:
    """The exception: the extension rounds gradient * upstream once, the native route the
    gradient of upstream 1 and then its product with upstream (_torch_ops.scaled)."""
    for key, (values, _) in extension.items():
        if not scaled_case(key):
            continue
        got_values, got = native["outputs"][key]
        assert got_values == values, key
        dtype = getattr(torch, key.split("-")[1])
        if dtype in _torch_ops.HALF:
            continue
        unit = _gradient(extension[key.removesuffix(str(cases.SCALE)) + "1.0"][1], dtype)
        rows = (cases.N,) if key.startswith("ce_none") else ()
        upstream = torch.full(rows, cases.SCALE, dtype=_torch_ops.value_dtype(dtype))
        expected = _torch_ops.scaled(unit, upstream, dtype)
        assert torch.equal(_gradient(got, dtype).view(torch.uint8), expected.view(torch.uint8)), key


@pytest.mark.parametrize(
    "dtype, margin, multiplier",
    [
        (torch.bfloat16, 110.0, 2.0**60),
        (torch.float16, 26.0, 2.0**20),
    ],
)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nonunit_cotangent_recovers_a_rounded_tail(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    margin: float,
    multiplier: float,
    reduction: L.Reduction,
) -> None:
    monkeypatch.setattr(_torch_ops, "eager", lambda: None)
    z = torch.tensor([[margin, 0], [0, margin]], dtype=dtype, requires_grad=True)
    y = torch.tensor([0, 1])
    value = L.cross_entropy(z, y, reduction=reduction)
    unit = torch.autograd.grad(value.sum(), z, retain_graph=True)[0]
    assert torch.count_nonzero(unit) == 0
    scale = (
        torch.tensor([-multiplier, multiplier]) if reduction == "none" else torch.tensor(multiplier)
    )
    got = torch.autograd.grad(value, z, scale)[0]
    tail = torch.sigmoid(torch.tensor(-margin, dtype=torch.float64))
    signs = torch.tensor([[-1, 1], [1, -1]], dtype=torch.float64)
    factor = scale.reshape(-1, 1).double() / (2 if reduction == "mean" else 1)
    expected = (signs * tail * factor).to(dtype)
    assert torch.count_nonzero(expected) == z.numel()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("classes", [7, 257])
def test_amplified_bfloat_multilabel_tail(classes: int) -> None:
    from masked_cases import amplified_multilabel_tail

    scores, target, upstream, reference = amplified_multilabel_tail(classes)
    z = torch.tensor(scores, dtype=torch.bfloat16, requires_grad=True)
    values = L.multilabel_cross_entropy(z, torch.tensor(target), backend="kernel", reduction="none")
    expected = torch.tensor(reference).to(z.dtype)
    for sign in (1, -1):
        grad = torch.autograd.grad(values, z, sign * torch.tensor(upstream), retain_graph=True)[0]
        torch.testing.assert_close(grad[:, 1], sign * expected, atol=0, rtol=0)
        assert not torch.signbit(grad[7::8, 1]).any()


def test_amplified_bfloat_multilabel_work_and_storage_zeros() -> None:
    # The first tail underflows FP32. The second is nonzero in FP32 and rounds away in BF16.
    z = torch.tensor([[87.5, -87.5, 0.0], [66.5, -66.5, 0.0]], dtype=torch.bfloat16)
    z.requires_grad_()
    mask = torch.tensor([[False, False, True], [False, False, True]])
    values = L.multilabel_cross_entropy(z, mask, reduction="none", backend="kernel")
    grad = torch.autograd.grad(values, z, torch.full_like(values, -(2.0**56)))[0]
    assert torch.count_nonzero(grad[:, 1]) == 0
    assert torch.signbit(grad[:, 1]).tolist() == [False, True]


def test_scaled_half_gradient_under_transforms() -> None:
    z = torch.tensor([[110, 0], [0, 110]], dtype=torch.bfloat16)
    y = torch.tensor([0, 1])
    scale = 2.0**60

    def loss(x: torch.Tensor) -> torch.Tensor:
        return scale * L.cross_entropy(x, y)

    tail = torch.sigmoid(torch.tensor(-110.0, dtype=torch.float64)) * (scale / 2)
    expected = (tail * torch.tensor([[-1, 1], [1, -1]])).to(z.dtype)
    batch = z.repeat(2, 1, 1)
    gradient = torch.func.grad(lambda x: torch.vmap(loss)(x).sum())(batch)
    torch.testing.assert_close(gradient, expected.expand_as(batch), rtol=0, atol=0)
    z.requires_grad_()
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(torch.autograd.grad(compiled(z), z)[0], expected, rtol=0, atol=0)


def test_repeated_matrix_calls_use_the_current_inputs_and_gradient_requirements() -> None:
    y = torch.tensor([0, 1, 2])
    z = torch.arange(15, dtype=torch.float64).reshape(5, 3).T
    L.cross_entropy(z.contiguous(), y, reduction="none")
    y = torch.tensor([2, 0, 1, 0, 0, 0])[::2]
    z = z.requires_grad_()
    actual = L.cross_entropy(z, y, reduction="none")
    expected = torch.nn.functional.cross_entropy(z, y, reduction="none")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), z)[0], torch.autograd.grad(expected.sum(), z)[0]
    )
    with pytest.raises(IndexError, match="Target 5 is out of bounds"):
        L.cross_entropy(z, torch.tensor([0, 1, 5]), reduction="none")

    p = z.detach().softmax(1)
    L.cross_entropy(z, p)
    p.requires_grad_()
    actual = L.cross_entropy(z, p)
    expected = torch.nn.functional.cross_entropy(z, p)
    torch.testing.assert_close(
        torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_backward_takes_the_saved_gradient(dtype: torch.dtype) -> None:
    """The native route's autograd (_torch_ops.Loss in eager mode): the logits' gradient is
    the forward's buffer (not a copy), a kept graph gives it again, and create_graph links it
    to the logits so that differentiating it raises, as with the extension."""
    reason = _torch_ops.cpu_kernels(0)
    if reason is not None:
        pytest.skip(reason)
    rows = Rows(KINDS["cross_entropy"])
    z, y = torch.randn(6, 5, dtype=dtype, requires_grad=True), torch.randint(0, 5, (6,))
    value, gradient = _torch_ops.Loss.apply(z, y, None, rows, _torch_ops.REDUCTIONS["mean"], True)
    address = gradient.data_ptr()
    del gradient
    value.backward()
    assert z.grad is not None and z.grad.data_ptr() == address
    value, _ = _torch_ops.Loss.apply(z * 1, y, None, rows, _torch_ops.REDUCTIONS["sum"], True)
    first = torch.autograd.grad(value, z, retain_graph=True)[0]
    assert torch.equal(first, torch.autograd.grad(value, z)[0])
    value, _ = _torch_ops.Loss.apply(z * 1, y, None, rows, _torch_ops.REDUCTIONS["mean"], True)
    (grad,) = torch.autograd.grad(value, z, create_graph=True)
    with pytest.raises(RuntimeError, match="^libsdca kernels compute first derivatives only"):
        grad.sum().backward()


def test_compiled_label_head_preserves_live_scores_and_retained_gradients() -> None:
    features = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 17
    weights = torch.linspace(-1.5, 1, 20).reshape(5, 4).requires_grad_()
    labels = torch.arange(12) % 5

    def objective(w: torch.Tensor, reuse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        scores = features @ w.T
        return L.cross_entropy(scores, labels, label_smoothing=0.1, reuse_logits=reuse), scores

    compiled = torch.compile(objective, backend="aot_eager", fullgraph=True)
    active, scores = compiled(weights, True)
    reference, original = compiled(weights, False)
    torch.testing.assert_close(active, reference, rtol=0, atol=0)
    for factor in (1.0, -0.75, 0.0, 1.0):
        u = torch.tensor(factor)
        (actual,) = torch.autograd.grad(active, weights, u, retain_graph=True)
        (expected,) = torch.autograd.grad(reference, weights, u, retain_graph=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scores, original, rtol=0, atol=0)


def test_compiled_score_origin_is_local_to_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from libsdca import _torch_ops

    _check_compiled_score_origins(True)
    monkeypatch.setattr(_torch_ops, "comptime", None)
    _check_compiled_score_origins(False)


def _check_compiled_score_origins(recognize: bool) -> None:
    from torch.fx import GraphModule

    operators: list[str] = []

    def capture(graph: GraphModule, _: Any) -> Any:
        operators.extend(
            str(node.target)
            for module in graph.modules()
            if isinstance(module, GraphModule)
            for node in module.graph.nodes
        )
        return graph.forward

    z = torch.ones(6, 5, requires_grad=True) * 2
    labels = torch.arange(6) % 5

    def losses(scores: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return L.cross_entropy(scores, labels) + L.cross_entropy(scores @ transform, labels)

    compiled = torch.compile(losses, backend=capture, fullgraph=True)
    compiled(z, torch.eye(5)).backward()
    assert {"libsdca.loss", "libsdca.loss.default"}.intersection(operators)
    assert bool({"libsdca.loss_", "libsdca.loss_.default"}.intersection(operators)) == recognize


def test_switched_off_the_native_route_is_silent(native: dict[str, Any]) -> None:
    assert native["explain"] == (
        "kernel",
        "the CPU kernels of the native library (float32 work; SDCA_CPU_EXTENSION=off)",
    )
    assert native["warnings"] == []


def test_without_a_compiler_the_native_route_warns_once(tmp_path: Path) -> None:
    found = child(tmp_path, "none", CXX="/nonexistent/c++", SDCA_CPU_EXTENSION="auto")
    (message,) = found["warnings"]
    assert message.startswith(f"libsdca: {NO_COMPILER}")
    assert "native library (libsdca_c)" in message and "SDCA_CPU_EXTENSION=off" in message
    backend, reason = found["explain"]
    assert backend == "kernel"
    assert reason.startswith(f"the CPU kernels of the native library (float32 work; {NO_COMPILER}")


@pytest.mark.parametrize("recompute", (True, False))
def test_compiled_chunked_linear_head(recompute: bool) -> None:
    hidden = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 23
    weights = torch.linspace(-1.5, 1, 20).reshape(5, 4).requires_grad_()
    labels = torch.arange(12) % 5

    def loss(w: torch.Tensor) -> torch.Tensor:
        return L.linear_cross_entropy(hidden, w, labels, chunk_bytes=100, recompute=recompute)

    reference = loss(weights)
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)(weights)
    torch.testing.assert_close(compiled, reference, rtol=0, atol=0)
    upstream = torch.tensor(-0.75)
    (actual,) = torch.autograd.grad(compiled, weights, upstream)
    (expected,) = torch.autograd.grad(reference, weights, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

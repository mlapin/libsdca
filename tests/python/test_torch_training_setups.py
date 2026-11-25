"""Training lifetimes of the direct BF16 distillation loss, including CUDA graph replay."""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from libsdca.torch import cross_entropy, distillation_loss
from torch.multiprocessing.spawn import spawn
from torch.utils.checkpoint import checkpoint


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


def inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(4817)
    scores = torch.randn(8, 4097, generator=gen, device=device, dtype=torch.bfloat16)
    teacher = torch.randn_like(scores)
    return scores.requires_grad_(), teacher


def loss(scores: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return distillation_loss(scores, teacher, temperature=2, backend="kernel")


def check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def reference(scores: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = scores.detach().clone().requires_grad_()
    result = loss(z, teacher)
    grad = torch.autograd.grad(result, z)[0]
    native = (
        -(F.softmax(teacher.float() / 2, dim=-1) * F.log_softmax(z.float() / 2, dim=-1))
        .sum(-1)
        .mean()
    )
    torch.testing.assert_close(result, native, rtol=1e-6, atol=1e-6)
    native_grad = torch.autograd.grad(native, z)[0]
    torch.testing.assert_close(grad.float(), native_grad.float(), rtol=0.01, atol=1e-8)
    return result.detach(), grad


@pytest.mark.parametrize("reentrant", [True, False])
def test_checkpoint_modes(device: str, reentrant: bool) -> None:
    x, t = inputs(device)
    expected, grad = reference(x, t)
    alias = x.detach()
    original = x.detach().clone()
    seen = []
    modes = []

    def call(z: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        modes.append(torch.is_grad_enabled())
        value = loss(z, teacher)
        seen.append((torch.is_grad_enabled(), value.detach()))
        return value

    v = checkpoint(call, x, t, use_reentrant=reentrant)
    check(v, expected)
    v.backward()
    assert x.grad is not None
    check(x.grad, grad)
    check(alias, original)
    assert modes == [not reentrant, True]
    for _, value in seen:
        check(value, expected)


def test_saved_tensor_offload_retained_vjp_and_accumulation(
    device: str, record_property: Callable[[str, object], None]
) -> None:
    x, t = inputs(device)
    originals = [x.detach().clone(), t.clone()]
    expected, grad = reference(x, t)
    stored: list[tuple[int, int]] = []
    restored = []
    cached_rows: list[weakref.ReferenceType[torch.Tensor]] = []

    def pack(a: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
        stored.append((a.data_ptr(), a.numel() * a.element_size()))
        host = a.detach().to("cpu", copy=True)
        if a.shape == (x.shape[0], 6) and a.dtype == torch.float64:
            cached_rows.append(weakref.ref(host))
        return a.device, host

    def unpack(saved: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
        target, host = saved
        restored.append(host.numel())
        return host.to(target)

    with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        primal = loss(x, t)
    check(primal, expected)
    assert not stored
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        first = loss(x, t)
        second = loss(x, t)
    check(first, expected)
    check(second, expected)
    extra = sum(n for ptr, n in stored if ptr not in {x.data_ptr(), t.data_ptr()})
    record_property("two_forward_extra_saved_bytes", extra)
    record_property("input_bytes", x.numel() * x.element_size() + t.numel() * t.element_size())
    if device == "cuda":
        # The offload hook must receive the six FP64 fields of the streamed row state for every row of both forwards.
        assert extra == 2 * x.shape[0] * 6 * 8
    a = torch.autograd.grad(first, x, first.new_tensor(-2), retain_graph=True)[0]
    if device == "cuda":
        assert len(cached_rows) == 2 and all(ref() is not None for ref in cached_rows)
    b = torch.autograd.grad(first, x, first.new_tensor(2))[0]
    check(a, -b)
    with pytest.raises(RuntimeError, match="second time|freed"):
        torch.autograd.grad(first, x)
    if device == "cuda":
        assert cached_rows[0]() is None and cached_rows[1]() is not None
    assert restored
    second.backward()
    if device == "cuda":
        assert cached_rows[1]() is None
    assert x.grad is not None
    check(x.grad, grad)
    check(x.detach(), originals[0])
    check(t, originals[1])
    # Accumulate independent graph contributions into a common parameter.
    x.grad = None
    v1, v2 = loss(x, t), loss(x, t)
    v1.backward()
    v2.backward()
    assert x.grad is not None
    check(x.grad, grad + grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fullgraph_compiled_and_cuda_graph_replay() -> None:
    x, t = inputs("cuda")
    fn = torch.compile(loss, fullgraph=True, mode="reduce-overhead")
    expected, grad = reference(x, t)
    try:
        value = fn(x, t)
        check(value, expected)
        check(torch.autograd.grad(value, x)[0], grad)
    finally:
        torch._dynamo.reset()
    # A real capture rejects host synchronization, including data-dependent reads.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # A leaf's AccumulateGrad node remembers the stream of its first use.
        # Keep capture operands separate from the compiled default-stream check.
        static_x = x.detach().clone().requires_grad_()
        static_t = t.clone()
        for _ in range(3):
            torch.autograd.grad(loss(static_x, static_t), static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = loss(static_x, static_t)
        captured_grad = torch.autograd.grad(captured, static_x)[0]
    for teacher in (t.clone(), x.detach().clone()):
        static_t.copy_(teacher)
        value, grad = reference(x, teacher)
        graph.replay()
        check(captured, value)
        check(captured_grad, grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("wrapper", ["ddp", "fsdp2"])
def test_distributed_head_gives_local_bfloat_logits(tmp_path: Path, wrapper: str) -> None:
    import torch.distributed as dist

    if not dist.is_nccl_available():
        pytest.skip("NCCL required")
    fsdp: Any = pytest.importorskip("torch.distributed.fsdp")
    if wrapper == "fsdp2" and not hasattr(fsdp, "fully_shard"):
        pytest.skip("FSDP2 required")
    dist.init_process_group("nccl", init_method=(tmp_path / "group").as_uri(), world_size=1, rank=0)
    try:
        torch.manual_seed(29)
        layer = torch.nn.Linear(16, 67, bias=False).cuda()
        plain = copy.deepcopy(layer)
        model: Any = layer
        if wrapper == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(
                layer, device_ids=[torch.cuda.current_device()]
            )
        else:
            fsdp.fully_shard(
                layer,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
        features = torch.randn(8, 16, device="cuda")
        target = torch.randn(8, 67, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(features)
            expected = loss(plain(features), target)
        assert logits.dtype == torch.bfloat16 and type(logits) is torch.Tensor
        actual = loss(logits, target)
        expected.backward()
        actual.backward()
        check(actual, expected)
        g = layer.weight.grad
        assert g is not None and plain.weight.grad is not None
        full_tensor = getattr(g, "full_tensor", None)
        check(full_tensor() if full_tensor is not None else g, plain.weight.grad)
    finally:
        dist.destroy_process_group()


def test_higher_derivative_contract_and_second_reference(device: str) -> None:
    x, t = inputs(device)
    intermediate = x + 0.0
    second_reference = intermediate.detach()
    original = intermediate.detach().clone()
    result = loss(intermediate, t)
    (first,) = torch.autograd.grad(result, x, create_graph=True)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad((first.float() * x.detach().float()).sum(), x)
    check(second_reference, original)
    x2 = x.detach().float().requires_grad_()
    composed = distillation_loss(x2, t.float(), temperature=2, backend="composed")
    (g,) = torch.autograd.grad(composed, x2, create_graph=True)
    (h,) = torch.autograd.grad((g * x2.detach()).sum(), x2)
    assert torch.isfinite(h).all()


def _two_rank_head(rank: int, rendezvous: str, wrapper: str, backend: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2)
    try:
        _compare_two_rank_head(rank, wrapper, device)
    finally:
        dist.destroy_process_group()


def _compare_two_rank_head(rank: int, wrapper: str, device: torch.device) -> None:
    torch.manual_seed(19)
    layer = torch.nn.Linear(16, 67, bias=False).to(device)
    reference_layer = copy.deepcopy(layer)
    model: Any = layer
    if wrapper == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            layer, device_ids=[rank] if device.type == "cuda" else None
        )
    else:
        import torch.distributed.fsdp as fsdp
        from torch.distributed.device_mesh import init_device_mesh

        fsdp.fully_shard(
            layer,
            mesh=init_device_mesh(device.type, (2,)),
            mp_policy=fsdp.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )

    # Each rank independently computes both single-batch gradients without collectives.
    gradients = []
    for member in range(2):
        gen = torch.Generator().manual_seed(67 + member)
        x = torch.randn(8, 16, generator=gen).to(device)
        teacher = torch.randn(8, 67, generator=gen).to(device, dtype=torch.bfloat16)
        with torch.autocast(device.type, torch.bfloat16):
            plain = reference_layer(x)
        assert plain.dtype == torch.bfloat16
        expected = loss(plain, teacher)
        gradients.append(torch.autograd.grad(expected, reference_layer.weight)[0])
        if rank == member:
            with torch.autocast(device.type, torch.bfloat16):
                local = model(x)
            assert type(local) is torch.Tensor and local.device == device
            assert local.dtype == torch.bfloat16
            actual = loss(local, teacher)
            check(actual, expected)
            actual.backward()
    assert not torch.equal(gradients[0], gradients[1])
    assert layer.weight.grad is not None
    full = getattr(layer.weight.grad, "full_tensor", None)
    if wrapper == "fsdp2":
        from torch.distributed.tensor import DTensor

        assert isinstance(layer.weight.grad, DTensor)
        assert layer.weight.grad.to_local().numel() < layer.weight.numel()
    actual_gradient = full() if full is not None else layer.weight.grad
    assert actual_gradient.dtype == torch.float32
    check(actual_gradient, (gradients[0] + gradients[1]) * 0.5)


@pytest.mark.parametrize("wrapper", ["ddp", "fsdp2"])
@pytest.mark.parametrize("backend", ["gloo", pytest.param("nccl", marks=pytest.mark.cuda)])
def test_two_rank_head_with_different_batches(tmp_path: Path, wrapper: str, backend: str) -> None:
    import torch.distributed as dist

    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo required")
    if backend == "nccl" and (
        not dist.is_nccl_available()
        or torch.cuda.device_count() < 2
        or any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(2))
    ):
        pytest.skip("two native BF16 CUDA devices and NCCL required")
    if wrapper == "fsdp2" and not hasattr(
        pytest.importorskip("torch.distributed.fsdp"), "fully_shard"
    ):
        pytest.skip("FSDP2 required")
    spawn(
        _two_rank_head, args=((tmp_path / "group").as_uri(), wrapper, backend), nprocs=2, join=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_weighted_mean_offloads_its_cuda_divisor() -> None:
    x = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3).requires_grad_()
    target = torch.tensor([0, 1, 2, 1, 0, 2, 1], device="cuda")
    weight = torch.tensor([0.5, 2.0, 3.25], device="cuda")
    expected = cross_entropy(x, target, weight=weight, backend="kernel")
    (grad,) = torch.autograd.grad(expected, x)
    scalars = []

    def pack(a: torch.Tensor) -> torch.Tensor:
        host = a.detach().cpu()
        if a.numel() == 1 and a.dtype == torch.float64:
            scalars.append(host.item())
        return host

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda a: a.to(x.device)):
        actual = cross_entropy(x, target, weight=weight, backend="kernel")
    check(actual, expected)
    assert float(weight[target].sum()) in scalars
    (replayed,) = torch.autograd.grad(actual, x, actual.new_tensor(-1.0))
    check(replayed, -grad)

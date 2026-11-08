"""Global DTensor losses and ordinary two-rank DDP gradients on CPU."""

import contextlib
import copy
import datetime
import importlib.util
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


@contextlib.contextmanager
def case(**labels: object) -> Iterator[None]:
    """Name the failing case in a worker process, where pytest subtests do not report."""
    try:
        yield
    except AssertionError as error:
        raise AssertionError(f"{labels}: {error}") from error


def distributed_worker(rank: int, rendezvous: str, task: str, device: str) -> None:
    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=180 if device == "cuda" else 45),
    )
    try:
        if task == "dtensor":
            check_dtensor_losses(device)
            check_dtensor_mesh_edges(device)
        else:
            check_ddp_gradients(rank)
    finally:
        dist.destroy_process_group()


def check_dtensor_losses(device: str = "cpu") -> None:
    from libsdca import LOSSES
    from libsdca.torch import (
        cross_entropy,
        distillation_loss,
        explain_backend,
        proximal_distillation_loss,
        sdca_loss,
    )
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    mesh = init_device_mesh(device, (dist.get_world_size(),))
    scores = torch.tensor(
        [
            [0.2, -0.5, 1.0, 0.8],
            [1.2, 0.3, -0.2, 0.1],
            [-0.3, 0.8, 0.4, -0.6],
            [0.5, -1.0, 0.3, 0.2],
        ],
        device=device,
    )
    labels = torch.tensor([0, 1, 2, 3], device=device)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, False],
            [True, False, False, True],
            [False, True, False, True],
        ],
        device=device,
    )
    precisions = [
        (torch.float32, "float32"),
        (torch.float32, "float64"),
        (torch.float64, "float64"),
    ]
    if device == "cuda":
        precisions += [
            (dtype, precision)
            for dtype in (torch.float16, torch.bfloat16)
            for precision in ("float32", "float64")
        ]
    last: tuple[torch.Tensor, torch.Tensor] | None = None
    for dtype, precision in precisions:
        logits = scores.to(dtype)
        weights = torch.tensor([0.2, 2.0, 0.0, 0.7], dtype=dtype, device=device)
        teacher = logits.flip(-1) + 0.13
        cases = [(loss, mask if loss.startswith("multilabel") else labels) for loss in LOSSES]
        cases += [
            ("entropy_mask", mask),
            ("distillation", teacher),
            ("proximal_distillation", teacher),
            ("weighted_ce", torch.tensor([0, -1, 1, 3], device=device)),
        ]
        for placement in (Replicate(), Shard(0), Shard(1)):
            for reduction in ("none", "sum", "mean"):
                for name, target in cases:
                    backends = (
                        ("torch", "native")
                        if device == "cpu"
                        and name not in ("distillation", "proximal_distillation", "weighted_ce")
                        else ("torch",)
                    )
                    for backend in backends:
                        with case(
                            dtype=dtype,
                            precision=precision,
                            placement=placement,
                            reduction=reduction,
                            loss=name,
                            backend=backend,
                        ):
                            local = logits.detach().clone().requires_grad_()
                            z = distribute_tensor(
                                logits.detach().clone().requires_grad_(), mesh, [placement]
                            )
                            target_placement = (
                                placement
                                if target.ndim == 2
                                else (Shard(0) if placement == Shard(0) else Replicate())
                            )
                            y = distribute_tensor(target.clone(), mesh, [target_placement])
                            last = z, y
                            options: dict[str, Any] = dict(reduction=reduction, precision=precision)
                            if name == "distillation":
                                expected = distillation_loss(
                                    local, target, temperature=0.7, **options
                                )
                                actual = distillation_loss(z, y, temperature=0.7, **options)
                            elif name == "proximal_distillation":
                                expected = proximal_distillation_loss(
                                    local, target, 0.7, 0.4, **options
                                )
                                actual = proximal_distillation_loss(z, y, 0.7, 0.4, **options)
                            elif name == "weighted_ce":
                                weight = distribute_tensor(weights.clone(), mesh, [Shard(0)])
                                expected = cross_entropy(
                                    local, target, weights, ignore_index=-1, **options
                                )
                                actual = cross_entropy(z, y, weight, ignore_index=-1, **options)
                            else:
                                objective = "cross_entropy" if name == "entropy_mask" else name
                                options.update(backend=backend, k=2, gamma=0.6, alpha=0.9)
                                expected = sdca_loss(local, target, objective, **options)
                                actual = sdca_loss(z, y, objective, **options)
                            assert isinstance(actual, DTensor), type(actual)
                            assert actual.placements == (Replicate(),)
                            tolerance = {
                                torch.float16: 2e-3,
                                torch.bfloat16: 2e-2,
                                torch.float32: 2e-5,
                                torch.float64: 1e-11,
                            }[dtype]
                            torch.testing.assert_close(
                                actual.full_tensor(),
                                expected,
                                rtol=tolerance,
                                atol=tolerance * 1e-2,
                            )
                            upstream = (
                                torch.tensor(
                                    [0.3, 0.7, 1.1, 0.9], dtype=expected.dtype, device=device
                                )
                                if reduction == "none"
                                else torch.tensor(1.7, dtype=expected.dtype, device=device)
                            )
                            expected.backward(upstream)
                            actual.backward(distribute_tensor(upstream, mesh, [Replicate()]))
                            grad = z.grad
                            assert isinstance(grad, DTensor), type(grad)
                            assert grad.placements == z.placements
                            torch.testing.assert_close(
                                grad.full_tensor(),
                                local.grad,
                                rtol=tolerance,
                                atol=tolerance * 1e-2,
                            )
            assert last is not None
            z, y = last
            assert explain_backend(z, y)[0] == "dtensor"
            for options in (
                {"loss": "unknown"},
                {"backend": "unknown"},
                {"precision": "mixed"},
            ):
                with pytest.raises(ValueError):
                    explain_backend(z, y, **options)

    # Ordinary tensor companions represent global replicated values. This is
    # useful for fixed labels or teachers without a separate DTensor wrapper.
    z = distribute_tensor(scores.double().requires_grad_(), mesh, [Shard(1)])
    value = sdca_loss(z, labels)
    assert isinstance(value, DTensor), type(value)
    torch.testing.assert_close(value.full_tensor(), sdca_loss(scores.double(), labels))
    other_mesh = DeviceMesh(device, torch.arange(dist.get_world_size()).reshape(1, -1))
    other_labels = distribute_tensor(labels, other_mesh, [Replicate(), Replicate()])
    with pytest.raises(ValueError, match="same device mesh"):
        sdca_loss(z, other_labels)


def check_dtensor_mesh_edges(device: str) -> None:
    """Unequal row/class shards and 2-D meshes retain a global mean denominator."""
    from libsdca.torch import cross_entropy, sdca_loss
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    size = dist.get_world_size()
    scores = torch.arange(25, dtype=torch.float64, device=device).reshape(5, 5).sin()
    labels = torch.tensor([0, 2, 4, 1, 3], device=device)
    ignored = torch.tensor([0, -1, 4, 1, 3], device=device)
    weights = torch.tensor([0.2, 2.0, 0.7, 0.0, 1.3], dtype=torch.float64, device=device)
    for shape in ((size,), (size, 1), (1, size)):
        mesh = DeviceMesh(device, torch.arange(size).reshape(shape))
        active = shape.index(size)
        for dimension in (0, 1):
            placements = [Replicate()] * len(shape)
            placements[active] = Shard(dimension)
            target_placements = [Replicate()] * len(shape)
            if dimension == 0:
                target_placements[active] = Shard(0)
            weight_placements = [Replicate()] * len(shape)
            weight_placements[active] = Shard(0)
            for objective in ("cross_entropy", "topk_cross_entropy", "weighted_ce"):
                with case(mesh=shape, shard=dimension, objective=objective):
                    local = scores.clone().requires_grad_()
                    z = distribute_tensor(scores.clone().requires_grad_(), mesh, placements)
                    target = ignored if objective == "weighted_ce" else labels
                    y = distribute_tensor(target, mesh, target_placements)
                    if objective == "weighted_ce":
                        w = distribute_tensor(weights, mesh, weight_placements)
                        expected = cross_entropy(local, target, weights, ignore_index=-1)
                        actual = cross_entropy(z, y, w, ignore_index=-1)
                    else:
                        expected = sdca_loss(local, target, objective, k=2)
                        actual = sdca_loss(z, y, objective, k=2)
                    assert isinstance(actual, DTensor), type(actual)
                    assert actual.placements == tuple(Replicate() for _ in shape)
                    torch.testing.assert_close(
                        actual.full_tensor(), expected, rtol=1e-11, atol=1e-13
                    )
                    expected.backward()
                    actual.backward()
                    grad = z.grad
                    assert isinstance(grad, DTensor), type(grad)
                    torch.testing.assert_close(
                        grad.full_tensor(), local.grad, rtol=1e-11, atol=1e-13
                    )


def check_ddp_gradients(rank: int) -> None:
    from libsdca.torch import cross_entropy, sdca_loss
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel

    x = torch.tensor(
        [[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0], [2.0, 1.0, -1.0], [0.2, -0.3, 0.8]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 3, 1, 2])
    model = torch.nn.Linear(3, 4, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 7 - 1)
        model.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1], dtype=torch.float64))
    reference = copy.deepcopy(model)
    F.cross_entropy(reference(x), labels).backward()
    local = slice(2 * rank, 2 * rank + 2)
    for loss, backend in ((cross_entropy, "torch"), (sdca_loss, "torch"), (sdca_loss, "native")):
        ddp = DistributedDataParallel(copy.deepcopy(model))
        loss(ddp(x[local]), labels[local], backend=backend).backward()
        for actual, expected in zip(ddp.module.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-14)


# DTensor's public mesh module (PyTorch >= 2.2).
DEVICE_MESH = (
    dist.is_available() and importlib.util.find_spec("torch.distributed.device_mesh") is not None
)


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="Gloo is unavailable"
)
class DistributedLossTests:
    def run_distributed(self, task: str, device: str = "cpu") -> None:
        with tempfile.TemporaryDirectory(prefix="libsdca-distributed-") as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            spawn(distributed_worker, args=(rendezvous, task, device), nprocs=2, join=True)

    @pytest.mark.slow
    @pytest.mark.skipif(not DEVICE_MESH, reason="torch.distributed.device_mesh needs PyTorch 2.2")
    def test_dtensor_global_values_and_gradients_all_objectives(self):
        self.run_distributed("dtensor")

    def test_ddp_parameter_gradients_match_full_batch_reference(self):
        self.run_distributed("ddp")


class CudaDistributedLossTests:
    @pytest.mark.skipif(
        not (dist.is_available() and dist.is_nccl_available() and torch.cuda.device_count() >= 2),
        reason="Two CUDA devices and NCCL are required",
    )
    def test_dtensor_global_values_and_gradients_all_objectives(self):
        DistributedLossTests().run_distributed("dtensor", "cuda")

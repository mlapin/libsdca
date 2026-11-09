"""Two-rank Hessian-vector products and first-order DTensor boundaries."""

import datetime
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from test_distributed import case
from torch.multiprocessing.spawn import spawn


def higher_worker(rank: int, rendezvous: str):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        check_higher_derivatives()
    finally:
        dist.destroy_process_group()


def check_higher_derivatives():
    from libsdca.torch import cross_entropy, distillation_loss, sdca_loss
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.nn import functional as F

    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    scores = torch.tensor(
        [[0.2, -0.4, 0.7], [1.0, 0.0, -0.1], [-0.3, 0.8, 0.4], [0.5, -1.0, 0.3]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9], [0.8, -0.6, -0.2], [0.5, 0.4, -0.3]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 2, 2, 1])
    ignored = torch.tensor([0, -1, 2, 1])
    weights = torch.tensor([0.5, 2.0, 1.5], dtype=torch.float64)
    for placement in (Shard(0), Shard(1)):
        v = distribute_tensor(direction, mesh, [placement])
        target_placement = Shard(0) if placement == Shard(0) else Replicate()
        distributed_labels = distribute_tensor(ignored, mesh, [target_placement])
        distributed_weights = distribute_tensor(weights, mesh, [Shard(0)])
        for objective in ("cross_entropy", "topk_cross_entropy", "weighted_ce"):
            with case(shard=placement, objective=objective):
                local = scores.clone().requires_grad_()
                z = distribute_tensor(scores.clone().requires_grad_(), mesh, [placement])
                if objective == "weighted_ce":
                    reference = F.cross_entropy(local, ignored, weights, ignore_index=-1)
                    actual = cross_entropy(
                        z,
                        distributed_labels,
                        distributed_weights,
                        ignore_index=-1,
                        backend="torch",
                    )
                else:
                    reference = F.cross_entropy(local, labels)
                    # Plain labels contain the same GLOBAL values on each rank.
                    actual = sdca_loss(z, labels, objective, k=1, backend="torch")
                (expected_gradient,) = torch.autograd.grad(reference, local, create_graph=True)
                (expected_hvp,) = torch.autograd.grad(expected_gradient, local, direction)
                (gradient,) = torch.autograd.grad(actual, z, create_graph=True)
                (hvp,) = torch.autograd.grad(gradient, z, v)
                assert isinstance(actual, DTensor) and isinstance(gradient, DTensor)
                assert isinstance(hvp, DTensor)
                torch.testing.assert_close(actual.full_tensor(), reference, rtol=1e-12, atol=1e-14)
                torch.testing.assert_close(
                    gradient.full_tensor(), expected_gradient, rtol=1e-11, atol=1e-13
                )
                torch.testing.assert_close(hvp.full_tensor(), expected_hvp, rtol=1e-11, atol=1e-13)
                assert hvp.placements == (placement,)

        # Distillation promises a stable first derivative, not a Hessian.
        # Wrapping it must preserve the explicit double-backward error.
        z = distribute_tensor(scores.clone().requires_grad_(), mesh, [placement])
        teacher = distribute_tensor(scores.flip(-1), mesh, [placement])
        (gradient,) = torch.autograd.grad(distillation_loss(z, teacher), z, create_graph=True)
        with pytest.raises(RuntimeError, match="first derivatives only"):
            torch.autograd.grad(gradient, z, v)


# DTensor's double backward through full_tensor reaches the local logits from PyTorch 2.12.
TORCH = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="Gloo is unavailable"
)
@pytest.mark.skipif(TORCH < (2, 12), reason="DTensor higher derivatives need PyTorch 2.12")
class DistributedHigherDerivativeTests:
    def test_global_hessian_vector_products_and_first_order_boundary(self):
        with tempfile.TemporaryDirectory(prefix="libsdca-dtensor-higher-") as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            spawn(higher_worker, args=(rendezvous,), nprocs=2, join=True)

"""Autograd-preserving full-input fallback for DTensor loss arguments."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor


def _dtensor_type() -> type[DTensor] | None:
    # Creating a DTensor loads its package; ordinary local calls need not
    # import the distributed stack. Keep the legacy name for older PyTorch.
    for name in ("torch.distributed.tensor", "torch.distributed._tensor"):
        dtensor = getattr(sys.modules.get(name), "DTensor", None)
        if dtensor is not None:
            return cast("type[DTensor]", dtensor)
    return None


def has_dtensor(*values: object) -> bool:
    # Ordinary tensors have no placements. Avoid importing or inspecting the
    # distributed class on this hot path; subclasses still get the full check.
    for value in values:
        if value is not None and type(value) is not torch.Tensor:
            dtensor = _dtensor_type()
            return dtensor is not None and any(isinstance(item, dtensor) for item in values)
    return False


def dtensor_mesh(*values: object) -> DeviceMesh:
    """Require one common mesh before issuing any input collectives."""
    dtensor = _dtensor_type()
    distributed = [value for value in values if dtensor is not None and isinstance(value, dtensor)]
    if not distributed:
        raise ValueError("At least one DTensor argument is required")
    mesh = distributed[0].device_mesh
    if any(value.device_mesh != mesh for value in distributed[1:]):
        raise ValueError("DTensor loss arguments must use the same device mesh")
    return mesh


def call_with_dtensors(
    function: Callable[..., torch.Tensor], /, *args: object, **kwargs: object
) -> torch.Tensor:
    """Gather full inputs differentiably and return the global loss replicated.

    Every rank computes the same complete loss, including global reductions.
    Local tensor companions must already contain identical global values on
    every rank. This fallback retains full inputs and intermediates per rank;
    it is not a memory-saving class-sharded kernel.
    """
    dtensor = _dtensor_type()
    assert dtensor is not None
    mesh = dtensor_mesh(*args, *kwargs.values())
    try:
        from torch.distributed.tensor import Replicate
    except ImportError:
        from torch.distributed._tensor.placement_types import Replicate

    def gather(value: object) -> object:
        # All ranks compute a complete global gradient, not partial sums.
        return value.full_tensor() if isinstance(value, dtensor) else value

    result = function(
        *(gather(value) for value in args),
        **{name: gather(value) for name, value in kwargs.items()},
    )
    # from_local normalizes incoming gradients to Replicate before returning
    # the local complete gradient; full_tensor then restores the input layout.
    return dtensor.from_local(result, mesh, [Replicate()] * mesh.ndim, run_check=False)

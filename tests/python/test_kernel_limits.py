"""The fused CUDA kernels' shape limits: 2**25 classes and 2**28 rows (TORCH.md).

Inputs at a limit take the fused kernels; one row or class more takes the tensor
operations. Logits this large do not fit in memory here, so the dispatch predicates get
stand-ins with the attributes they read and a stub for the (CUDA-only) kernel module.
"""

import importlib.util
import types
from unittest import mock

import pytest
import torch
from libsdca import _cross_entropy, _cuda_launch, _entropy, _multilabel
from libsdca import torch as sdca_torch

ROWS, CLASSES = 2**28, 2**25
# A stand-in kernel module: what the predicates read of it.
KERNEL = types.SimpleNamespace(CLASSES=CLASSES, covers=lambda *args: True)


def logits(rows: int, classes: int, dtype: torch.dtype = torch.float16) -> types.SimpleNamespace:
    """A CUDA logits stand-in of shape (rows, classes)."""
    return types.SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        dtype=dtype,
        shape=(rows, classes),
        numel=lambda: rows * classes,
        contiguous=lambda: None,
    )


def target(rows: int, dtype: torch.dtype = torch.int64, shape: tuple[int, ...] = ()):
    return types.SimpleNamespace(dtype=dtype, shape=shape or (rows,), numel=lambda: rows)


# (rows, classes, fused): at and just past each limit.
CASES = [
    (ROWS, 2, True),
    (ROWS + 1, 2, False),
    (1, CLASSES, True),
    (1, CLASSES + 1, False),
    (ROWS, CLASSES, True),
]


def test_the_documented_limits():
    assert _cuda_launch.ROWS == ROWS
    assert _entropy.CLASSES == CLASSES


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="needs Triton")
def test_the_kernels_share_the_limits():
    from libsdca import _triton_entropy, _triton_weighted_entropy

    assert _triton_entropy.CLASSES == _entropy.CLASSES
    assert _triton_weighted_entropy.CLASSES == _entropy.CLASSES


def test_integer_entropy(subtests: pytest.Subtests):
    with mock.patch.object(_entropy, "_kernel", lambda: KERNEL):
        for rows, classes, fused in CASES:
            with subtests.test(rows=rows, classes=classes):
                z = logits(rows, classes)
                assert _entropy.covers(z.device, rows, classes) == fused
                assert _entropy.fused(z) == fused  # pyright: ignore[reportArgumentType]
                # precision="float32" (explicit) takes the fused float32 entropy or float64.
                path = sdca_torch._indexed_entropy_path(
                    z,  # pyright: ignore[reportArgumentType]
                    target(rows),  # pyright: ignore[reportArgumentType]
                    "cross_entropy",
                    1,
                    explicit=True,
                )
                assert path == fused
    # Without the kernel (no Triton or no NVIDIA GPU) nothing is fused.
    with mock.patch.object(_entropy, "_kernel", lambda: None):
        assert not _entropy.fused(logits(1, 2))  # pyright: ignore[reportArgumentType]


def test_weighted_entropy(subtests: pytest.Subtests):
    with (
        mock.patch.object(_entropy, "_kernel", lambda: KERNEL),
        mock.patch.object(_cross_entropy, "_weighted_kernel", lambda: KERNEL),
    ):
        for rows, classes, fused in CASES:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with subtests.test(rows=rows, classes=classes, dtype=dtype):
                    z = logits(rows, classes, dtype)
                    can = _cross_entropy._can_fuse_weighted(
                        z,  # pyright: ignore[reportArgumentType]
                        target(rows),  # pyright: ignore[reportArgumentType]
                        -100,
                        1,
                    )
                    assert can == fused


def test_boolean_masks(subtests: pytest.Subtests):
    # Rows only: the class limit (67,107,840) is the multilabel kernels' own (below).
    with mock.patch.object(_multilabel, "_kernel", lambda: KERNEL):
        for rows, fused in ((ROWS, True), (ROWS + 1, False)):
            for objective in _multilabel.OBJECTIVES:
                with subtests.test(rows=rows, objective=objective):
                    z = logits(rows, 3, torch.float64)
                    mask = target(rows, torch.bool, z.shape)
                    applies = _multilabel.applies(
                        objective,
                        z,  # pyright: ignore[reportArgumentType]
                        mask,  # pyright: ignore[reportArgumentType]
                        3,
                        "auto",
                        None,
                        "mean",
                    )
                    assert applies == fused


def test_distillation_rows(subtests: pytest.Subtests):
    calls = []

    def kernel(proximal: bool = False):
        return lambda *args: calls.append(args) or "fused"

    with mock.patch.object(sdca_torch, "_kernel_distill", kernel):
        for rows, fused in ((ROWS, True), (ROWS + 1, False)):
            with subtests.test(rows=rows):
                s = logits(rows, 3)
                result = sdca_torch._fused_distillation(
                    s,  # pyright: ignore[reportArgumentType]
                    s,  # pyright: ignore[reportArgumentType]
                    2.0,
                    False,
                    "mean",
                )
                assert (result == "fused") == fused
    assert len(calls) == 1


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="needs Triton")
def test_triton_row_and_class_bounds():
    from libsdca import _triton_hinge, _triton_multilabel

    assert _triton_hinge.covers(logits(ROWS, 2))  # pyright: ignore[reportArgumentType]
    assert not _triton_hinge.covers(logits(ROWS + 1, 2))  # pyright: ignore[reportArgumentType]
    # The multilabel backward grid: fewer than 65,536 column blocks of 1,024.
    assert _triton_multilabel.covers(67_107_840)
    assert not _triton_multilabel.covers(67_107_841)


def test_proximal_entropy_rows():
    # One program per row: fewer than 2**31 rows (the grid limit), any class count.
    with mock.patch.object(sdca_torch, "_kernel_prox", lambda: "kernel"):
        assert sdca_torch._prox_kernel(logits(2**31 - 1, CLASSES + 1)) == "kernel"  # pyright: ignore[reportArgumentType]
        assert sdca_torch._prox_kernel(logits(2**31, 2)) is None  # pyright: ignore[reportArgumentType]

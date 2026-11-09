"""Full-range row statistics for chunked linear CE reductions.

Retain four scalars per row, so reduction can recover overflowing row losses,
weighted means with overflowing denominators, and individually vanished tails.
The common 16/32-bit path does not need this extra work.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from . import _cross_entropy, _entropy


class FullRange:
    """Global reduction metadata; GEMM logits and gradients remain chunked."""

    def __init__(self, weights: Tensor, reduction: str, ignore: int) -> None:
        self.weights, self.reduction, self.ignore = weights, reduction, ignore
        self.saved = weights.new_zeros((4, weights.numel()))
        largest = weights.max()
        self.normalizer = (largest, (weights / torch.where(largest > 0, largest, 1)).sum())

    def statistics(self, kernel: int, z: Tensor, y: Tensor, rows: Tensor, raw: Any):
        """Decode existing row statistics, with a tensor fallback for non-double logits."""
        n = z.shape[0]
        if kernel == 0:
            return tuple(t.masked_fill(rows.isnan(), math.nan) for t in raw[1])
        if kernel == 2:
            maximum, label, total = raw[: 3 * n].view(n, 3).unbind(1)
            log_sum = total.log()
        elif kernel == 1:
            maximum, _, total, _, _ = raw[: 5 * n].view(n, 5).unbind(1)
            label = z.gather(1, y.clamp(0, z.shape[1] - 1)[:, None]).squeeze(1).double()
            log_sum = total.log()
        elif z.dtype == torch.float64:
            maximum, log_sum, label, _ = raw[: 4 * n].view(n, 4).unbind(1)
        else:
            return _entropy._statistics(z, y.clamp(0, z.shape[1] - 1), rows.isnan())
        maximum = maximum.masked_fill(rows.isnan(), math.nan)
        return maximum, label, log_sum, (maximum - label) + log_sum

    def save(self, part: slice | Tensor, stats: tuple[Tensor, ...]) -> None:
        self.saved[:, part] = torch.stack(stats)

    def value(self) -> Tensor:
        return _cross_entropy._weighted_value(
            tuple(self.saved.unbind(0)), self.weights, self.reduction
        )

    def gradient(
        self,
        z: Tensor,
        y: Tensor,
        stats: tuple[Tensor, ...],
        part: slice | Tensor,
        scale: Tensor,
        poison: Tensor | None = None,
    ):
        """The chunk's gradient; NaN everywhere where the 0-d poison holds (a NaN weight makes
        every row of a reduction NaN)."""
        weights = self.weights[part]
        if poison is not None:
            weights = weights.masked_fill(poison, math.nan)
        result = _cross_entropy._weighted_gradient(
            z,
            y,
            stats,
            weights,
            scale,
            self.reduction,
            normalizer=self.normalizer if self.reduction == "mean" else None,
        )
        if self.reduction == "mean":
            undefined = (y != self.ignore) & (self.normalizer[0] == 0)
            result = result.masked_fill(undefined[:, None], math.nan)
        return result

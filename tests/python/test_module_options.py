"""SDCALoss modules validate options at construction and on assignment."""

from typing import Any

import pytest
import torch
from libsdca.torch import MulticlassHingeLoss, SDCALoss, multiclass_hinge_loss


class TestModuleOptions:
    def test_construction_rejects_bad_options(self, subtests: pytest.Subtests):
        cases: list[dict[str, Any]] = [
            {"loss": "bogus"},
            {"gamma": -1.0},
            {"backend": "x"},
            {"precision": "x"},
        ]
        for options in cases:
            with subtests.test(options=options), pytest.raises(ValueError):
                SDCALoss(**options)

    def test_rejected_assignment_keeps_the_module(self):
        z, y = torch.randn(8, 5, dtype=torch.float64), torch.randint(0, 5, (8,))
        criterion = MulticlassHingeLoss(gamma=0.5)
        expected = criterion(z, y)
        with pytest.raises(ValueError):
            criterion.gamma = -1.0
        assert criterion.gamma == 0.5
        torch.testing.assert_close(criterion(z, y), expected)
        criterion.gamma = 0.0
        torch.testing.assert_close(criterion(z, y), multiclass_hinge_loss(z, y))

"""Loss modules validate their options at construction, and each call checks the current
attributes: an assigned option takes effect, a bad one raises at the next call."""

from typing import Any

import pytest
import torch
from libsdca.torch import (
    CrossEntropyLoss,
    DistillationLoss,
    LinearCrossEntropyLoss,
    MulticlassHingeLoss,
    MultilabelCrossEntropyLoss,
    MultilabelHingeLoss,
    ProximalCrossEntropyLoss,
    ProximalDistillationLoss,
    TopKCrossEntropyLoss,
    TopKHingeLoss,
    multiclass_hinge_loss,
)


class TestModuleOptions:
    def test_construction_rejects_bad_options(self, subtests: pytest.Subtests):
        cases: list[tuple[type[torch.nn.Module], dict[str, Any], type[Exception]]] = [
            (MulticlassHingeLoss, {"gamma": -1.0}, ValueError),
            (MulticlassHingeLoss, {"gamma": float("nan")}, ValueError),
            (MulticlassHingeLoss, {"gamma": "0.5"}, TypeError),
            (TopKHingeLoss, {"k": 0}, ValueError),
            (TopKHingeLoss, {"k": 2, "variant": "gamma"}, ValueError),
            (TopKCrossEntropyLoss, {"k": 2.0}, TypeError),
            (ProximalCrossEntropyLoss, {"alpha": 0.0}, ValueError),
            (ProximalCrossEntropyLoss, {"ignore_index": 1.5}, TypeError),
            (ProximalCrossEntropyLoss, {"weight": [1.0, 2.0]}, TypeError),
            (ProximalCrossEntropyLoss, {"weight": torch.ones(2, 2)}, ValueError),
            (MultilabelHingeLoss, {"backend": "x"}, ValueError),
            (MultilabelCrossEntropyLoss, {"backend": 1}, TypeError),
            (MultilabelCrossEntropyLoss, {"precision": "x"}, ValueError),
            (MultilabelCrossEntropyLoss, {"precision": 1.0}, TypeError),
            (MultilabelCrossEntropyLoss, {"reduction": "avg"}, ValueError),
            (MultilabelCrossEntropyLoss, {"dim": 1.0}, TypeError),
            (DistillationLoss, {"temperature": 0.0}, ValueError),
            (ProximalDistillationLoss, {"alpha": -1.0}, ValueError),
            (CrossEntropyLoss, {"label_smoothing": 1.5}, ValueError),
            (CrossEntropyLoss, {"backend": "cuda"}, ValueError),
            (CrossEntropyLoss, {"precision": "double"}, ValueError),
            (LinearCrossEntropyLoss, {"chunk_bytes": 0}, ValueError),
            (LinearCrossEntropyLoss, {"recompute": 1}, TypeError),
        ]
        for module, options, error in cases:
            with subtests.test(module=module.__name__, options=options), pytest.raises(error):
                module(**options)

    def test_assignment_takes_effect_and_is_checked_by_the_call(self):
        z, y = torch.randn(8, 5, dtype=torch.float64), torch.randint(0, 5, (8,))
        criterion = MulticlassHingeLoss(gamma=0.5)
        expected = criterion(z, y)
        criterion.gamma = -1.0
        with pytest.raises(ValueError, match="gamma"):
            criterion(z, y)
        criterion.gamma = 0.5
        torch.testing.assert_close(criterion(z, y), expected)
        criterion.gamma = 0.0
        torch.testing.assert_close(criterion(z, y), multiclass_hinge_loss(z, y))
        hinge = TopKHingeLoss(k=2)
        hinge.variant = "alpha"
        hinge.k = 1
        torch.testing.assert_close(hinge(z, y), multiclass_hinge_loss(z, y))

    def test_modules_hold_their_own_options_only(self):
        assert not hasattr(TopKHingeLoss(k=2), "alpha")
        assert not hasattr(MulticlassHingeLoss(), "k")
        assert not hasattr(ProximalCrossEntropyLoss(), "gamma")
        assert not hasattr(MultilabelCrossEntropyLoss(), "weight")
        for module in (TopKHingeLoss(k=2), ProximalCrossEntropyLoss(), DistillationLoss()):
            # A plain nn.Module: assigning another loss's name is just an attribute.
            module.loss = "cross_entropy"  # pyright: ignore[reportArgumentType]
            assert type(module).__setattr__ is torch.nn.Module.__setattr__

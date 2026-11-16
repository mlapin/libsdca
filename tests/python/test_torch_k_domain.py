"""The top-k PyTorch losses check 1 <= k < classes on every backend: k < 1 when the function
or module is given it, k >= classes with the logits (docs/API.md, parameter domains). k is
required and keyword-only; the other losses take no k."""

import pytest
import torch
from libsdca import torch as sdca_torch

CLASSES = 5
BELOW = [0, -1]
ABOVE = [CLASSES, 2 * CLASSES]
VALID = [1, CLASSES - 1]
BACKENDS = ["auto", "native", "torch"]
# The top-k losses: function, module and fixed options.
TOPK = {
    "topk_cross_entropy": (sdca_torch.topk_cross_entropy, sdca_torch.TopKCrossEntropyLoss, {}),
    "topk_hinge_alpha": (
        sdca_torch.topk_hinge_loss,
        sdca_torch.TopKHingeLoss,
        {"variant": "alpha"},
    ),
    "topk_hinge_beta": (sdca_torch.topk_hinge_loss, sdca_torch.TopKHingeLoss, {"variant": "beta"}),
}


def devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def problem(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn(6, CLASSES, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    return z.to(device), (torch.arange(6) % CLASSES).to(device)


def cases() -> list[tuple[str, str, str]]:
    return [
        (loss, device, backend)
        for loss in TOPK
        for device in devices()
        for backend in BACKENDS
        if device == "cpu" or backend != "native"
    ]


def test_functions_and_modules(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        z, target = problem(device)
        function, module, fixed = TOPK[loss]
        for k in BELOW:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    function(z, target, k=k, backend=backend, **fixed)
                with pytest.raises(ValueError, match="k must be"):
                    module(k=k, backend=backend, **fixed)
        for k in ABOVE:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    function(z, target, k=k, backend=backend, **fixed)
                # Constructed without classes: forward checks k.
                criterion = module(k=k, backend=backend, **fixed)
                with pytest.raises(ValueError, match="k must be"):
                    criterion(z, target)
        for k in VALID:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                value = function(z, target, k=k, backend=backend, **fixed)
                assert torch.isfinite(value).all()
                torch.testing.assert_close(module(k=k, backend=backend, **fixed)(z, target), value)


def test_k_is_required_keyword_only_and_an_integer():
    z, target = problem("cpu")
    for function, module, fixed in TOPK.values():
        with pytest.raises(TypeError):
            function(z, target, **fixed)
        with pytest.raises(TypeError):
            module(**fixed)
        with pytest.raises(TypeError):
            function(z, target, 2, **fixed)
        for k in (2.0, True, "2"):
            with pytest.raises(TypeError, match="k must be an integer"):
                function(z, target, k=k, **fixed)
            with pytest.raises(TypeError, match="k must be an integer"):
                module(k=k, **fixed)


def test_explain_backend(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        z, target = problem(device)
        function, module, fixed = TOPK[loss]
        for k in BELOW + ABOVE:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    sdca_torch.explain_backend(function, z, target, k=k, backend=backend, **fixed)
        with subtests.test(loss=loss, device=device, backend=backend, k=CLASSES - 1):
            k = CLASSES - 1
            explained = sdca_torch.explain_backend(
                function, z, target, k=k, backend=backend, **fixed
            )
            criterion = module(k=k, backend=backend, **fixed)
            assert sdca_torch.explain_backend(criterion, z, target) == explained


def test_losses_without_k_reject_it():
    z, target = problem("cpu")
    mask = torch.nn.functional.one_hot(target, CLASSES).bool()
    mask[:, (target + 1) % CLASSES] = False
    for function, y in (
        (sdca_torch.cross_entropy, target),
        (sdca_torch.multiclass_hinge_loss, target),
        (sdca_torch.proximal_cross_entropy, target),
        (sdca_torch.multilabel_cross_entropy, mask),
        (sdca_torch.multilabel_hinge_loss, mask),
    ):
        with pytest.raises(TypeError):
            function(z, y, k=2)  # pyright: ignore[reportCallIssue]
    for module in (
        sdca_torch.MulticlassHingeLoss,
        sdca_torch.ProximalCrossEntropyLoss,
        sdca_torch.MultilabelCrossEntropyLoss,
        sdca_torch.MultilabelHingeLoss,
        sdca_torch.CrossEntropyLoss,
    ):
        with pytest.raises(TypeError):
            module(k=2)  # pyright: ignore[reportCallIssue]

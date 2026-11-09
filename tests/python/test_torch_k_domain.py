"""Every PyTorch entry point checks 1 <= k < classes for every loss and backend, also the
losses that do not use k (docs/API.md, parameter domains)."""

import libsdca
import pytest
import torch
from libsdca import torch as sdca_torch

CLASSES = 5
OUT_OF_RANGE = [0, CLASSES, 2 * CLASSES]
VALID = [1, CLASSES - 1]
BACKENDS = ["auto", "native", "torch"]
# The entry points that take k, besides sdca_loss and SDCALoss.
FUNCTIONS = {
    "topk_cross_entropy": sdca_torch.topk_cross_entropy,
    "topk_hinge_alpha": sdca_torch.topk_hinge_alpha_loss,
    "topk_hinge_beta": sdca_torch.topk_hinge_beta_loss,
}
MODULES = {
    "topk_cross_entropy": sdca_torch.TopKCrossEntropyLoss,
    "topk_hinge_alpha": sdca_torch.TopKHingeAlphaLoss,
    "topk_hinge_beta": sdca_torch.TopKHingeBetaLoss,
}


def devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def problem(loss: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn(6, CLASSES, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    y = torch.arange(6) % CLASSES
    target = torch.nn.functional.one_hot(y, CLASSES).bool() if "multilabel" in loss else y
    return z.to(device), target.to(device)


def cases() -> list[tuple[str, str, str]]:
    return [
        (loss, device, backend)
        for loss in libsdca.LOSSES
        for device in devices()
        for backend in BACKENDS
        if device == "cpu" or backend != "native"
    ]


def test_sdca_loss(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        z, target = problem(loss, device)
        for k in OUT_OF_RANGE:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    sdca_torch.sdca_loss(z, target, loss, backend=backend, k=k)
        for k in VALID:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                value = sdca_torch.sdca_loss(z, target, loss, backend=backend, k=k)
                assert torch.isfinite(value).all()


def test_module(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        z, target = problem(loss, device)
        for k in OUT_OF_RANGE:
            # Constructed without classes: forward checks k.
            criterion = sdca_torch.SDCALoss(loss, backend=backend, k=k)
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    criterion(z, target)
        for k in VALID:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                assert torch.isfinite(sdca_torch.SDCALoss(loss, backend=backend, k=k)(z, target))


def test_functions_and_modules_that_take_k(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        if loss not in FUNCTIONS:
            continue
        z, target = problem(loss, device)
        function, module = FUNCTIONS[loss], MODULES[loss]
        for k in OUT_OF_RANGE:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    function(z, target, k, backend=backend)
                with pytest.raises(ValueError, match="k must be"):
                    module(k, backend=backend)(z, target)
        for k in VALID:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                expected = sdca_torch.sdca_loss(z, target, loss, backend=backend, k=k)
                torch.testing.assert_close(function(z, target, k, backend=backend), expected)
                torch.testing.assert_close(module(k, backend=backend)(z, target), expected)
    for device in devices():
        z, target = problem("topk_hinge_beta", device)
        for k in OUT_OF_RANGE:
            with subtests.test(loss="topk_hinge_loss", device=device, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    sdca_torch.topk_hinge_loss(z, target, k, variant="alpha")
                with pytest.raises(ValueError, match="k must be"):
                    sdca_torch.TopKHingeLoss(k)(z, target)


def test_explain_backend(subtests: pytest.Subtests):
    for loss, device, backend in cases():
        z, target = problem(loss, device)
        for k in OUT_OF_RANGE:
            with subtests.test(loss=loss, device=device, backend=backend, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    sdca_torch.explain_backend(z, target, loss, backend=backend, k=k)
        with subtests.test(loss=loss, device=device, backend=backend, k=CLASSES - 1):
            sdca_torch.explain_backend(z, target, loss, backend=backend, k=CLASSES - 1)


def test_multiclass_hinge_ignores_a_valid_k(subtests: pytest.Subtests):
    for _, device, backend in cases():
        z, target = problem("multiclass_hinge", device)
        for gamma in (0.0, 0.5):
            with subtests.test(device=device, backend=backend, gamma=gamma):
                options = {"backend": backend, "gamma": gamma}
                expected = sdca_torch.sdca_loss(z, target, "multiclass_hinge", **options)
                value = sdca_torch.sdca_loss(
                    z, target, "multiclass_hinge", k=CLASSES - 1, **options
                )
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

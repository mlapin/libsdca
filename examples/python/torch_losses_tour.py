"""Every libsdca PyTorch loss on one batch: module and functional forms, forward and backward."""

import libsdca.torch as L
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
n, m = 16, 10
logits = torch.randn(n, m, device=device, requires_grad=True)
labels = torch.randint(m, (n,), device=device)  # int64 class indices
mask = torch.rand(n, m, device=device) < 0.3  # boolean label sets
mask[:, 0], mask[:, 1] = True, False  # >= 1 positive and >= 1 negative per row
teacher = torch.randn(n, m, device=device)

# (name, module, functional, target)
cases = [
    ("cross_entropy", L.CrossEntropyLoss(), L.cross_entropy, labels),
    (
        "topk_cross_entropy",
        L.TopKCrossEntropyLoss(k=3),
        lambda z, t: L.topk_cross_entropy(z, t, k=3),
        labels,
    ),
    ("multiclass_hinge", L.MulticlassHingeLoss(), L.multiclass_hinge_loss, labels),
    (
        "multiclass_hinge gamma=0.5",
        L.MulticlassHingeLoss(gamma=0.5),
        lambda z, t: L.multiclass_hinge_loss(z, t, gamma=0.5),
        labels,
    ),
    (
        "topk_hinge_alpha",
        L.TopKHingeLoss(k=3, variant="alpha"),
        lambda z, t: L.topk_hinge_loss(z, t, k=3, variant="alpha"),
        labels,
    ),
    (
        "topk_hinge_beta",
        L.TopKHingeLoss(k=3, variant="beta"),
        lambda z, t: L.topk_hinge_loss(z, t, k=3, variant="beta"),
        labels,
    ),
    (
        "topk_hinge_alpha gamma=0.5",
        L.TopKHingeLoss(k=3, gamma=0.5, variant="alpha"),
        lambda z, t: L.topk_hinge_loss(z, t, k=3, gamma=0.5, variant="alpha"),
        labels,
    ),
    (
        "topk_hinge_beta gamma=0.5",
        L.TopKHingeLoss(k=3, gamma=0.5, variant="beta"),
        lambda z, t: L.topk_hinge_loss(z, t, k=3, gamma=0.5, variant="beta"),
        labels,
    ),
    ("multilabel_cross_entropy", L.MultilabelCrossEntropyLoss(), L.multilabel_cross_entropy, mask),
    (
        "multilabel_hinge gamma=0.5",
        L.MultilabelHingeLoss(gamma=0.5),
        lambda z, t: L.multilabel_hinge_loss(z, t, gamma=0.5),
        mask,
    ),
    (
        "proximal_cross_entropy",
        L.ProximalCrossEntropyLoss(alpha=0.5),
        lambda z, t: L.proximal_cross_entropy(z, t, alpha=0.5),
        labels,
    ),
    (
        "distillation",
        L.DistillationLoss(temperature=4.0),
        lambda z, t: L.distillation_loss(z, t, temperature=4.0),
        teacher,
    ),
    (
        "proximal_distillation",
        L.ProximalDistillationLoss(temperature=4.0, alpha=0.5),
        lambda z, t: L.proximal_distillation_loss(z, t, temperature=4.0, alpha=0.5),
        teacher,
    ),
]

for name, module, functional, target in cases:
    value = module(logits, target)
    (grad,) = torch.autograd.grad(value, logits)
    torch.testing.assert_close(functional(logits, target), value)
    assert torch.isfinite(grad).all()
    path = L.explain_backend(module, logits, target)[0]
    print(f"{name:28s} loss {value.item():8.4f}  |grad| {grad.norm().item():.4f}  ({path})")

# reduction="none" gives per-example losses; dim selects the class axis.
per_row = L.TopKHingeLoss(k=2, reduction="none")(logits, labels)
assert per_row.shape == (n,)
spatial = torch.randn(2, m, 4, 4, device=device)
spatial_labels = torch.randint(m, (2, 4, 4), device=device)
assert L.topk_hinge_loss(spatial, spatial_labels, k=2, dim=1).ndim == 0

# Class weights and an ignored label, with torch's weighted mean.
weight = torch.rand(m, device=device)
assert L.TopKHingeLoss(k=2, weight=weight, ignore_index=0)(logits, labels).isfinite()

# torch.func transforms: per-example gradients by vmap over grad.
per_example = torch.func.vmap(torch.func.grad(lambda z, y: L.topk_hinge_loss(z, y, k=2)))
assert per_example(logits.detach()[:, None], labels[:, None]).shape == (n, 1, m)

"""Float32 proximal_cross_entropy rows of many classes against the float64 solve.

A sequential float32 residual sum misplaces the normalizer by about C/100 epsilons at
C = 65536; every CPU path, and the streamed CUDA kernels, must stay within the 60 of
TORCH.md. The kernels sum within a tile in float32 and across tiles in float64; the
interpreter test runs them, and the streamed distillation kernels, over hundreds of tiles.
"""

import os
from unittest import mock

import interpreter
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import sdca_loss

BUDGET = 60
EPS = torch.finfo(torch.float32).eps
PATHS = {
    "cpu-tensor": ("torch", "off"),
    "native": ("native", "off"),
    **({"cpu-kernel": ("auto", "on")} if _cpu_entropy.buildable() else {}),
}


def solve(
    x: torch.Tensor, target: torch.Tensor, backend: str, flag: str, precision: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_()
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
        value = sdca_loss(
            x,
            target,
            "proximal_cross_entropy",
            backend=backend,
            reduction="none",
            precision=precision,
        )
        value.backward(torch.ones_like(value))
    assert x.grad is not None
    return value.detach().double().cpu(), x.grad.double().cpu()


def errors(
    value: torch.Tensor, grad: torch.Tensor, value64: torch.Tensor, grad64: torch.Tensor
) -> tuple[float, float]:
    """The value error relative to the value and the gradient error relative to its 1-norm,
    in float32 epsilons."""
    value, grad = value.double().cpu(), grad.double().cpu()
    value64, grad64 = value64.double().cpu(), grad64.double().cpu()
    grad_error = ((grad - grad64).abs().amax(-1) / grad64.abs().sum(-1)).max().item() / EPS
    return ((value - value64).abs() / value64.abs()).max().item() / EPS, grad_error


def rows(classes: int, n: int = 4, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Scores, labels and label masks (1% labels) of n rows."""
    generator = torch.Generator().manual_seed(seed)
    z = 3 * torch.randn(n, classes, generator=generator)
    labels = torch.randint(0, classes, (n,), generator=generator)
    mask = torch.rand(z.shape, generator=generator) < 0.01
    index = torch.arange(n)
    mask[index, labels], mask[index, (labels + 1) % classes] = True, False
    return z, labels, mask


# (loss, distillation rows): the rows are independent programs, those above 0 at an offset.
STREAMED = [
    ("proximal_cross_entropy", ()),
    *(
        (name, picked)
        for name in ("distillation_loss", "proximal_distillation_loss")
        for picked in ((0,), (1, 2))
    ),
]


def streamed_check(
    name: str, picked: tuple[int, ...] = (0, 1, 2)
) -> dict[str, tuple[float, float]]:
    """A streamed float32 kernel over 300 tiles of 16 columns against the float64 tensor
    solve, on random rows and on rows whose float32 total of 1 drops each later tile sum
    (student probabilities exp(-19.5), distillation row 0); its value and gradient errors,
    in float32 epsilons."""
    os.environ["SDCA_CPU_EXTENSION"] = "off"
    from libsdca import _tensor_prox, _triton_prox
    from libsdca import _triton_distillation as kernels
    from libsdca.torch import distillation_loss, proximal_distillation_loss

    classes, block = 4800, 16
    _triton_prox.REGISTER_CLASSES = _triton_prox._STREAM_BLOCK = block
    kernels._CHUNK = kernels._STREAM_BLOCK = block
    kernels._STREAM = kernels._BACKWARD = (block, 1)
    found = {}
    if name == "proximal_cross_entropy":
        z, labels, _ = rows(classes, 2, seed=2)
        positive = torch.nn.functional.one_hot(labels, classes).bool()
        value, grad = _triton_prox.proximal_entropy(z, positive, 1.0, True)
        expected = _tensor_prox.proximal_entropy(z.double(), positive, 1.0)
        found[name] = errors(value, grad, *expected)

    generator = torch.Generator().manual_seed(1)
    student = 2 * torch.randn(3, classes, generator=generator)
    student[0] = -19.5
    student[0, 0] = 0
    teacher = student + 0.5 * torch.randn(3, classes, generator=generator)
    teacher[0] = student[0] * 0.1
    student, teacher = student[list(picked)], teacher[list(picked)]
    for proximal in (False, True):
        if name != ("proximal_distillation_loss" if proximal else "distillation_loss"):
            continue
        x = student.double().requires_grad_()
        if proximal:
            value, grad = kernels.proximal_distillation(student, teacher, 1.0, 1.0, False)
            value64 = proximal_distillation_loss(x, teacher.double(), 1.0, 1.0, reduction="none")
        else:
            value, stats = kernels.forward(student, teacher, 1.0, False, "none")
            ones = torch.ones(len(picked))
            grad = kernels.backward(student, teacher, stats, ones, 1.0, False, "none")
            value64 = distillation_loss(x, teacher.double(), 1.0, reduction="none")
        value64.sum().backward()
        assert x.grad is not None
        found[name] = errors(value, grad, value64.detach(), x.grad)
    assert list(found) == [name], name
    print(found)
    # Float32 tile sums leave 21 epsilons in proximal_distillation_loss and 4 in
    # distillation_loss values here; float64 ones under one.
    for name, error in found.items():
        assert max(error) <= (BUDGET if name == "proximal_cross_entropy" else 2), (name, error)
    return found


class WideFloat32ProxTests:
    def test_gradient_one_norm(self, subtests: pytest.Subtests):
        z, labels, mask = rows(65536)
        for kind, target in (("label", labels), ("mask", mask)):
            value64, grad64 = solve(z.double(), target, "torch", "0", None)
            for name, (backend, flag) in PATHS.items():
                with subtests.test(target=kind, path=name):
                    value, grad = solve(z, target, backend, flag, "float32")
                    value_error, grad_error = errors(value, grad, value64, grad64)
                    assert grad_error <= BUDGET, "gradient"
                    assert value_error <= BUDGET, "value"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_streamed_cuda_kernel(self, subtests: pytest.Subtests):
        for classes in (65536, 128256):
            z, labels, mask = rows(classes)
            for kind, target in (("label", labels), ("mask", mask)):
                value64, grad64 = solve(z.double(), target, "torch", "0", None)
                with subtests.test(classes=classes, target=kind):
                    value, grad = solve(z.cuda(), target.cuda(), "auto", "0", "float32")
                    value_error, grad_error = errors(value, grad, value64, grad64)
                    assert grad_error <= BUDGET, "gradient"
                    assert value_error <= BUDGET, "value"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("name", "picked"), STREAMED, ids=[n + "".join(f"-{i}" for i in p) for n, p in STREAMED]
    )
    def test_streamed_kernels_in_the_interpreter(self, name: str, picked: tuple[int, ...]):
        call = f"t.streamed_check({name!r}, {picked!r})"
        interpreter.run(f"import test_wide_float32_prox as t\n{call}\n")

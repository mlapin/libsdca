"""reuse_logits of libsdca.torch on the CPU: a backward that computes the gradient (the
steps here scale the upstream; a unit upstream takes the forward's gradient) writes it over
logits that nothing else references, else allocates a new one and warns (TORCH_WARN_ONCE);
False never reuses and never warns. Gradients equal the composed backend's in every case."""

from __future__ import annotations

import textwrap
import threading
import warnings
import weakref
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import libsdca.torch as L
import pytest
import torch
from child_process import run_python
from libsdca import _torch_ops
from torch.torch_version import TorchVersion

_NO_EXTENSION = _torch_ops.cpu_kernels() or _torch_ops.fallback(torch.device("cpu"))
if _NO_EXTENSION is not None:
    pytest.skip(f"no CPU extension: {_NO_EXTENSION}", allow_module_level=True)
if TorchVersion(torch.__version__) < "2.10":
    pytest.skip("logits reuse needs PyTorch 2.10 or later", allow_module_level=True)

MESSAGE = "libsdca: the logits are still referenced"
N, D, C = 6, 5, 7
UPSTREAM = 0.5  # the gradient of the loss in the steps, as in gradient accumulation
Net = torch.nn.Linear
Step = Callable[..., object]


def backward(loss: torch.Tensor, **options: Any) -> None:
    loss.backward(torch.full_like(loss, UPSTREAM), **options)


@pytest.fixture(autouse=True)
def warn_always() -> Iterator[None]:
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    try:
        yield
    finally:
        torch.set_warn_always(previous)


class Probe(torch.autograd.Function):
    """The identity into new storage, recording the storage of its output and of the gradient
    that reaches it (without holding either)."""

    seen: ClassVar[dict[str, int]] = {}

    @staticmethod
    def forward(ctx: Any, *inputs: torch.Tensor) -> torch.Tensor:
        out = inputs[0].clone()
        Probe.seen["logits"] = out.untyped_storage().data_ptr()
        return out

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        Probe.seen["grad"] = grads[0].untyped_storage().data_ptr()
        return grads[0]


def probe(h: torch.Tensor) -> torch.Tensor:
    return Probe.apply(h)  # type: ignore[no-any-return]


def _reused() -> bool:
    return Probe.seen["logits"] == Probe.seen["grad"]


def _net(dtype: torch.dtype = torch.float32) -> Net:
    torch.manual_seed(0)
    return Net(D, C, dtype=dtype)


def _data(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return torch.randn(N, D, generator=g, dtype=dtype), torch.randint(0, C, (N,), generator=g)


def _ours(record: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in record if MESSAGE in str(w.message)]


def _grad(net: Net) -> torch.Tensor:
    assert net.weight.grad is not None
    return net.weight.grad


def _backward(step: Step, **options: Any) -> tuple[torch.Tensor, int]:
    """The weight gradient of step(net, x, y, **options) and the number of reuse warnings."""
    net, (x, y) = _net(), _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        step(net, x, y, **options)
    return _grad(net), len(_ours(record))


def _composed(step: Step) -> torch.Tensor:
    return _backward(step, backend="composed")[0]


def inline(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> None:
    backward(L.cross_entropy(probe(net(x)), y, **options))


def deleted(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> None:
    logits = probe(net(x))
    loss = L.cross_entropy(logits, y, **options)
    del logits
    backward(loss)


def module(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> None:
    loss = L.CrossEntropyLoss(**options)(probe(net(x)), y)
    backward(loss)


def live(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> torch.Tensor:
    logits = probe(net(x))
    backward(L.cross_entropy(logits, y, **options))
    return logits


def view(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> torch.Tensor:
    logits = probe(net(x))
    loss = L.cross_entropy(logits, y, **options)
    head = logits[:2]
    del logits
    backward(loss)
    return head


def weak(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> object:
    logits = probe(net(x))
    loss = L.cross_entropy(logits, y, **options)
    ref = weakref.ref(logits)
    del logits
    backward(loss)
    return ref


def retained(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> None:
    backward(L.cross_entropy(probe(net(x)), y, **options), retain_graph=True)


def saved(net: Net, x: torch.Tensor, y: torch.Tensor, **options: Any) -> None:
    logits = probe(net(x))
    square = (logits * logits).mean()  # recorded first, so its backward runs after the loss's
    loss = L.cross_entropy(logits, y, **options) + square
    del logits
    backward(loss)


def _name(step: Step) -> str:
    return getattr(step, "__name__", "")


@pytest.mark.parametrize("step", [inline, deleted, module], ids=_name)
def test_reuses_dead_logits(step: Step):
    grad, warned = _backward(step)
    assert _reused() and warned == 0
    torch.testing.assert_close(grad, _composed(step))


def _not_reused(step: Step) -> bool:
    # After saved, the probe sees the sum of both gradients, allocated once the logits are
    # freed, which may take their address.
    return step is saved or not _reused()


@pytest.mark.parametrize("step", [live, view, weak, retained, saved], ids=_name)
def test_referenced_logits_warn(step: Step):
    grad, warned = _backward(step)
    assert _not_reused(step) and warned == 1
    torch.testing.assert_close(grad, _composed(step))


@pytest.mark.parametrize("step", [inline, live, view, weak, retained, saved], ids=_name)
def test_false_never_reuses_or_warns(step: Step):
    grad, warned = _backward(step, reuse_logits=False)
    assert _not_reused(step) and warned == 0
    torch.testing.assert_close(grad, _composed(step))


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_dtypes(dtype: torch.dtype):
    net, (x, y) = _net(dtype), _data(dtype)
    deleted(net, x, y)
    assert _reused()
    with pytest.warns(UserWarning, match=MESSAGE):
        logits = live(net, x, y)
    assert not _reused()
    del logits


# (function, target, options): every loss function of libsdca.torch but linear_cross_entropy.
LOSSES: list[tuple[Callable[..., torch.Tensor], str, dict[str, Any]]] = [
    (L.cross_entropy, "labels", {}),
    (L.topk_cross_entropy, "labels", {"k": 2}),
    (L.multiclass_hinge_loss, "labels", {}),
    (L.topk_hinge_loss, "labels", {"k": 2}),
    (L.proximal_cross_entropy, "labels", {}),
    (L.multilabel_cross_entropy, "mask", {}),
    (L.multilabel_hinge_loss, "mask", {}),
    (L.distillation_loss, "teacher", {"temperature": 2.0}),
    (L.proximal_distillation_loss, "teacher", {}),
]


def _target(kind: str, y: torch.Tensor) -> torch.Tensor:
    if kind == "labels":
        return y
    if kind == "mask":
        mask = torch.zeros(N, C, dtype=torch.bool)
        mask[torch.arange(N), y] = True
        mask[:, -1] = ~mask[:, -1] | (y == C - 1)
        mask[:, -2] = False
        return mask
    return torch.randn(N, C, generator=torch.Generator().manual_seed(2))


@pytest.mark.parametrize(("loss", "kind", "options"), LOSSES, ids=[f[0].__name__ for f in LOSSES])
@pytest.mark.parametrize("reuse", [True, False])
def test_every_loss(
    loss: Callable[..., torch.Tensor], kind: str, options: dict[str, Any], reuse: bool
):
    t = _target(kind, _data()[1])

    def step(net: Net, x: torch.Tensor, **more: Any) -> tuple[bool, torch.Tensor]:
        backward(loss(probe(net(x)), t, **options, **more))
        dead = (_reused(), _grad(net).clone())
        net.weight.grad = None
        logits = probe(net(x))
        backward(loss(logits, t, **options, **more))
        return dead

    net, (x, _) = _net(), _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        reused, grad = step(net, x, reuse_logits=reuse)
    assert reused == reuse and not _reused()
    assert len(_ours(record)) == int(reuse)
    composed = _net()
    want = step(composed, x, backend="composed")[1]
    torch.testing.assert_close(grad, want)
    torch.testing.assert_close(_grad(net), _grad(composed))


@pytest.mark.parametrize(("loss", "kind", "options"), LOSSES, ids=[f[0].__name__ for f in LOSSES])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_unit_upstream_takes_the_forward_gradient(
    loss: Callable[..., torch.Tensor], kind: str, options: dict[str, Any], reduction: str
):
    """The CPU forward of a mean or sum computes the gradient of an upstream of 1 in its pass:
    backward returns it (no reuse, no warning), equal to the gradient computed again."""
    t = _target(kind, _data()[1])
    net, (x, _) = _net(), _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loss(probe(net(x)), t, **options, reduction=reduction).backward()
    assert not _reused() and not _ours(record)
    taken = _grad(net).clone()
    net.weight.grad = None
    out = loss(probe(net(x)), t, **options, reduction=reduction, reuse_logits=False)
    out.backward(retain_graph=True)
    assert torch.equal(_grad(net), taken)
    net.weight.grad = None
    out.backward()  # the forward's gradient was taken: computed again
    assert torch.equal(_grad(net), taken)


@pytest.mark.parametrize(("loss", "kind", "options"), LOSSES, ids=[f[0].__name__ for f in LOSSES])
@pytest.mark.parametrize("shape", ["leaf", "view", "copied"])
def test_leaves_are_never_reused(
    loss: Callable[..., torch.Tensor], kind: str, options: dict[str, Any], shape: str
):
    """A leaf (or a view of one) is the caller's tensor: no reuse and no warning, even with
    no other reference."""
    t = _target(kind, _data()[1])
    data = torch.randn(N, C, generator=torch.Generator().manual_seed(3))
    if shape == "copied":  # classes on dim 1 of (N, C, 1): the loss copies them last
        data = data[:, :, None]
        t = t[:, None] if kind == "labels" else t[:, :, None]

    def grad(**more: Any) -> torch.Tensor:
        leaf = data.clone().requires_grad_()
        logits = leaf.view(N, C) if shape == "view" else leaf
        loss(logits, t, **options, **more).backward()
        assert torch.equal(leaf.detach(), data) and leaf.grad is not None
        return leaf.grad

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = grad()
    assert not _ours(record)
    torch.testing.assert_close(got, grad(backend="composed"))


def test_module_option():
    loss = L.TopKHingeLoss(k=2, reuse_logits=False)
    assert "reuse_logits=False" in repr(loss)
    net, (x, y) = _net(), _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        logits = probe(net(x))
        loss(logits, y).backward()
    assert not _reused() and not _ours(record)
    bad: Any = 1
    with pytest.raises(TypeError, match="reuse_logits must be True or False"):
        L.TopKHingeLoss(k=2, reuse_logits=bad)
    with pytest.raises(TypeError, match="reuse_logits must be True or False"):
        L.cross_entropy(logits, y, reuse_logits=bad)


def test_no_warning_without_reuse_path():
    """No warning where reuse is impossible anyway: the Python fallback under torch.func and
    torch.compile, gradients that build a graph, logits that need no gradient."""
    net, (x, y) = _net(), _data()
    logits = net(x)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        torch.func.grad(L.cross_entropy)(logits.detach(), y)
        compiled = torch.compile(L.cross_entropy, backend="aot_eager", fullgraph=True)
        compiled(logits, y).backward(retain_graph=True)
        torch.autograd.grad(L.cross_entropy(logits, y), logits, create_graph=True)
        with torch.no_grad():
            L.cross_entropy(logits, y)
        L.cross_entropy(logits.detach(), y)
    assert not _ours(record)


def test_filtered_and_threads():
    net, (x, y) = _net(), _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=MESSAGE)
        live(net, x, y)
    assert not _ours(record)
    logits = probe(net(x))
    loss = L.cross_entropy(logits, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        thread = threading.Thread(target=backward, args=(loss,))
        thread.start()
        thread.join()
    assert len(_ours(record)) == 1


def test_warns_once_per_process():
    code = textwrap.dedent(
        """
        import warnings
        import libsdca.torch as L
        import torch

        x, y = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                logits = torch.nn.functional.linear(x, torch.randn(3, 3, requires_grad=True))
                L.cross_entropy(logits, y).backward(torch.tensor(0.5))
        print(sum("logits are still referenced" in str(w.message) for w in record))
        """
    )
    done = run_python("-c", code, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[-1] == "1"

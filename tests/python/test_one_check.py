"""NumPy, PyTorch and JAX check loss parameters and options in one place (_frontend): the same
bad argument raises the same exception with the same message, but for the framework's words."""

from __future__ import annotations

import re
from typing import Any

import libsdca
import numpy as np
import pytest
from libsdca import _frontend as _f

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import libsdca.jax as J  # noqa: E402
import libsdca.torch as T  # noqa: E402

Z = np.arange(12, dtype=np.float64).reshape(3, 4) / 4
Y = np.array([0, 1, 2])

# (id, loss, its parameters, options); NumPy has no reduction.
CASES = [
    ("unknown parameter", "topk_cross_entropy", {"k": 2, "gamma": 1.0}, {}),
    ("parameter of another loss", "multiclass_hinge", {"temperature": 2.0}, {}),
    ("k out of range", "topk_hinge", {"k": 4}, {}),
    ("k zero", "topk_cross_entropy", {"k": 0}, {}),
    ("k float", "topk_hinge", {"k": 1.5}, {}),
    ("variant", "topk_hinge", {"k": 2, "variant": "gamma"}, {}),
    ("gamma negative", "multiclass_hinge", {"gamma": -1.0}, {}),
    ("step zero", "proximal_cross_entropy", {"gamma": 0.0}, {}),
    ("precision", "cross_entropy", {}, {"precision": "float16"}),
    ("precision type", "topk_cross_entropy", {"k": 2}, {"precision": 1}),
    ("reduction", "cross_entropy", {}, {"reduction": "max"}),
    ("reduction type", "multiclass_hinge", {}, {"reduction": 1}),
]


def _words(message: str) -> str:
    return re.sub(r"\b(np|torch|jnp)\.", "<fw>.", message)


def _raised(call: Any) -> tuple[type, str]:
    try:
        call()
    except (TypeError, ValueError) as error:
        return type(error), _words(str(error))
    raise AssertionError("no error")


def _function(lib: Any, name: str) -> Any:
    return getattr(lib, name, None) or getattr(lib, f"{name}_loss")


@pytest.mark.parametrize(
    ("name", "params", "options"), [case[1:] for case in CASES], ids=[case[0] for case in CASES]
)
def test_frameworks_raise_alike(name: str, params: dict, options: dict):
    found = {
        "torch": _raised(
            lambda: _function(T, name)(
                torch.from_numpy(Z), torch.from_numpy(Y), **params, **options
            )
        ),
        "jax": _raised(
            lambda: _function(J, name)(jnp.asarray(Z), jnp.asarray(Y), **params, **options)
        ),
    }
    if "reduction" not in options:
        found["numpy"] = _raised(
            lambda: libsdca.loss_and_grad(Z, Y, libsdca.LOSSES[name](**params), **options)
        )
    assert len(set(found.values())) == 1, found


def test_modules_raise_alike():
    found = {
        "numpy": _raised(lambda: libsdca.TopKHinge(2, gamma=0.5, temperature=2.0)),
        "torch": _raised(lambda: T.TopKHingeLoss(k=2, gamma=0.5, temperature=2.0)),
    }
    assert set(found.values()) == {(TypeError, "topk_hinge takes no parameter temperature")}


def test_targets_take_one_role():
    """Probability targets take a negative ignore_index (unused) and refuse others, as
    PyTorch, in both frameworks; the student's logits are named alike."""
    p = np.full((3, 4), 0.25)
    for value in (-100, -1):
        a = T.cross_entropy(torch.from_numpy(Z), torch.from_numpy(p), ignore_index=value)
        b = J.cross_entropy(jnp.asarray(Z), jnp.asarray(p), ignore_index=value)
        np.testing.assert_allclose(float(a), float(b), rtol=1e-6)
    found = {
        "torch": _raised(
            lambda: T.cross_entropy(torch.from_numpy(Z), torch.from_numpy(p), ignore_index=0)
        ),
        "jax": _raised(lambda: J.cross_entropy(jnp.asarray(Z), jnp.asarray(p), ignore_index=0)),
    }
    assert set(found.values()) == {(ValueError, _f.IGNORED_PROBABILITIES)}
    zi = Z.astype(np.int32)
    found = {
        "torch": _raised(lambda: T.distillation_loss(torch.from_numpy(zi), torch.from_numpy(Z))),
        "jax": _raised(lambda: J.distillation_loss(jnp.asarray(zi), jnp.asarray(Z))),
    }
    assert set(found.values()) == {(TypeError, "student logits must be floating-point, not int32")}

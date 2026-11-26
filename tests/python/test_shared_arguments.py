"""The arguments of libsdca.torch and libsdca.jax, validated in one place (_loss_options): the
same call with the same bad argument raises the same exception with the same message in both,
but for the framework's words (dim or axis, torch or jnp dtypes)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import libsdca.jax as J  # noqa: E402
import libsdca.torch as L  # noqa: E402
from named_losses import function, parameters  # noqa: E402

jax.config.update("jax_enable_x64", True)

LABELS = ("topk_cross_entropy", "multiclass_hinge", "topk_hinge_alpha", "proximal_cross_entropy")
MASKS = ("multilabel_cross_entropy", "multilabel_hinge")
DISTILLATION = ("distillation", "proximal_distillation")
Z = np.arange(12, dtype=np.float64).reshape(3, 4) / 4
Y = np.array([0, 1, 2])
M = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], bool)


def _call(
    framework: str, loss: str, z: Any, target: Any, params: dict | None = None, **options: Any
) -> Any:
    """The loss of numpy z and target in a framework; params replace the loss's own."""
    lib: Any = L if framework == "torch" else J
    convert = torch.from_numpy if framework == "torch" else jnp.asarray
    if "axis" in options and framework == "torch":
        options["dim"] = options.pop("axis")
    z, target = convert(np.asarray(z)), convert(np.asarray(target))
    if "weight" in options:
        options["weight"] = convert(np.asarray(options["weight"]))
    if loss in DISTILLATION:
        fn = lib.distillation_loss if loss == "distillation" else lib.proximal_distillation_loss
        return fn(z, target, **(params or {}), **options)
    own = parameters(loss, k=2) if params is None else {**parameters(loss, k=2), **params}
    return function(lib, loss)(z, target, **own, **options)


def _words(message: str) -> str:
    """A message with the framework's words replaced."""
    message = re.sub(r"\b(torch|jnp)\.", "<fw>.", message)
    message = re.sub(r"\b(dim|axis)\b", "<axis>", message)
    return message.replace("class dimension", "class <axis>").replace("class axis", "class <axis>")


def _target(loss: str) -> np.ndarray:
    return Z if loss in DISTILLATION else M if loss in MASKS else Y


# (id, loss names, logits, target or None for the loss's own, options)
CASES = [
    ("integer logits", LABELS + MASKS, Z.astype(np.int32), None, {}),
    ("0-d logits", DISTILLATION, np.float64(1.0), np.float64(1.0), {}),
    ("axis out of range", LABELS + MASKS + DISTILLATION, Z, None, {"axis": 3}),
    ("boolean labels", LABELS, Z, Y.astype(bool), {}),
    ("float labels", LABELS, Z, Y.astype(np.float64), {}),
    ("label shape", LABELS, Z, Y[:2], {}),
    ("integer mask", MASKS, Z, M.astype(np.int64), {}),
    ("mask shape", MASKS, Z, M[:2], {}),
    ("teacher shape", DISTILLATION, Z, Z[:, :3], {}),
    ("integer teacher", DISTILLATION, Z, Z.astype(np.int32), {}),
    ("integer weight", LABELS, Z, Y, {"weight": np.ones(4, np.int64)}),
    ("weight shape", LABELS, Z, Y, {"weight": np.ones(3)}),
    (
        "float32 precision of float64 logits",
        LABELS + MASKS + DISTILLATION,
        Z,
        None,
        {"precision": "float32"},
    ),  # fmt: skip
    ("float16 precision", LABELS + MASKS + DISTILLATION, Z, None, {"precision": "float16"}),
    ("reduction", LABELS + MASKS + DISTILLATION, Z, None, {"reduction": "max"}),
    ("backend native", LABELS + MASKS + DISTILLATION, Z, None, {"backend": "native"}),
    ("backend torch", LABELS + MASKS + DISTILLATION, Z, None, {"backend": "torch"}),
    ("reduction type", LABELS + MASKS + DISTILLATION, Z, None, {"reduction": 1}),
    ("ignore_index type", LABELS, Z, Y, {"ignore_index": 1.5}),
    ("temperature", DISTILLATION, Z, Z, {"temperature": 0.0}),
    ("temperature type", DISTILLATION, Z, Z, {"temperature": "1"}),
]
PARAMETERS = [  # (id, loss, parameters that replace its own)
    ("k zero", "topk_cross_entropy", {"k": 0}),
    ("k float", "topk_cross_entropy", {"k": 1.5}),
    ("k classes", "topk_hinge_alpha", {"k": 4, "variant": "alpha"}),
    ("gamma negative", "multiclass_hinge", {"gamma": -1.0}),
    ("gamma string", "multilabel_hinge", {"gamma": "0"}),
    ("variant", "topk_hinge_alpha", {"k": 2, "variant": "gamma"}),
    ("step zero", "proximal_cross_entropy", {"gamma": 0.0}),
    ("step nan", "proximal_distillation", {"gamma": float("nan")}),
]


def _raised(fn: Any) -> tuple[type, str]:
    try:
        fn()
    except (TypeError, ValueError) as error:
        return type(error), _words(str(error))
    raise AssertionError("no error")


@pytest.mark.parametrize(
    ("losses", "z", "target", "options"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_bad_arguments_raise_alike(losses: tuple[str, ...], z: Any, target: Any, options: dict):
    for loss in losses:
        t = _target(loss) if target is None else target
        found = {
            framework: _raised(lambda f=framework: _call(f, loss, z, t, **dict(options)))
            for framework in ("torch", "jax")
        }
        assert found["torch"] == found["jax"], loss


@pytest.mark.parametrize(
    ("loss", "given"), [p[1:] for p in PARAMETERS], ids=[p[0] for p in PARAMETERS]
)
def test_bad_parameters_raise_alike(loss: str, given: dict):
    found = {
        framework: _raised(lambda f=framework: _call(f, loss, Z, _target(loss), params=given))
        for framework in ("torch", "jax")
    }
    assert found["torch"] == found["jax"]


@pytest.mark.parametrize("backend", ("auto", "kernel", "composed"))
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_same_call_takes_the_same_backend(backend: str, dtype: type):
    """explain_backend of the same call names the same backend in both frameworks (on the
    CPU: libsdca's kernels or the framework's operations)."""
    for loss in LABELS + MASKS + DISTILLATION + ("cross_entropy",):
        z = Z.astype(dtype)
        found = {}
        for framework in ("torch", "jax"):
            lib: Any = L if framework == "torch" else J
            convert = torch.from_numpy if framework == "torch" else jnp.asarray
            t = _target(loss)
            target = convert(t.astype(dtype) if loss in DISTILLATION else t)
            if loss in DISTILLATION:
                fn = (
                    lib.distillation_loss
                    if loss == "distillation"
                    else lib.proximal_distillation_loss
                )
                options: dict[str, Any] = {}
            else:
                fn, options = function(lib, loss), parameters(loss, k=2)
            found[framework] = lib.explain_backend(
                fn, convert(z), target, backend=backend, **options
            )[0]
        assert found["torch"] == found["jax"], (loss, found)

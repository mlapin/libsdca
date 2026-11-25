"""The cases of test_native_route_torch.py and test_native_route_jax.py: every loss with its
options (class weights, ignore_index, label smoothing, probabilities, reductions, upstream
derivatives other than 1, and in JAX jit and vmap) in each dtype, as (values, gradient)
bytes. The tests compute them with the CPU extensions and in a child without them (the
native library): run() is the child.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from typing import Any

import numpy as np

N, C = 37, 11
DTYPES = ("float32", "float64", "bfloat16", "float16")
# An upstream derivative of the loss other than 1.
SCALE = 0.37
IGNORE = -100


def _inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    labels = rng.integers(0, C, N)
    labels[3] = IGNORE
    mask = rng.random((N, C)) < 0.3
    mask[:, 0], mask[:, 1] = True, False
    probabilities = rng.random((N, C))
    return dict(
        z=rng.standard_normal((N, C)),
        labels=labels,
        weight=rng.uniform(0.1, 2.0, C),
        mask=mask,
        teacher=rng.standard_normal((N, C)),
        probabilities=probabilities / probabilities.sum(1, keepdims=True),
    )


def _losses(m: Any, t: dict[str, Any]) -> dict[str, Any]:
    y, w = t["labels"], t["weight"]
    kept = t["kept"]
    return {
        "ce": lambda z: m.cross_entropy(z, kept),
        "ce_weight_smoothing": lambda z: m.cross_entropy(
            z, y, weight=w, ignore_index=IGNORE, label_smoothing=0.1
        ),
        "ce_sum": lambda z: m.cross_entropy(z, kept, reduction="sum"),
        "ce_none": lambda z: m.cross_entropy(z, kept, reduction="none").sum(),
        "ce_probabilities": lambda z: m.cross_entropy(z, t["probabilities"], label_smoothing=0.2),
        "topk_ce": lambda z: m.topk_cross_entropy(z, kept, k=3, weight=w),
        "hinge": lambda z: m.multiclass_hinge_loss(z, kept, gamma=0.5, weight=w),
        "topk_hinge": lambda z: m.topk_hinge_loss(z, y, k=2, variant="alpha", ignore_index=IGNORE),
        "multilabel_hinge": lambda z: m.multilabel_hinge_loss(z, t["mask"]),
        "multilabel_ce": lambda z: m.multilabel_cross_entropy(z, t["mask"]),
        "proximal_ce": lambda z: m.proximal_cross_entropy(z, kept, gamma=0.7),
        "distillation": lambda z: m.distillation_loss(z, t["teacher"], temperature=2.0),
        "proximal_distillation": lambda z: m.proximal_distillation_loss(
            z, t["teacher"], temperature=2.0, gamma=0.5
        ),
    }


def torch_outputs() -> dict[str, tuple[bytes, bytes]]:
    import libsdca.torch as L
    import torch

    out = {}
    for dtype in DTYPES:
        kind = getattr(torch, dtype)
        raw = _inputs(np.random.default_rng(0))
        t = {key: torch.tensor(a) for key, a in raw.items()}
        t["weight"] = t["weight"].to(kind)
        t["teacher"], t["probabilities"] = t["teacher"].to(kind), t["probabilities"].to(kind)
        t["kept"] = t["labels"].clamp(0)
        for name, loss in _losses(L, t).items():
            with torch.no_grad():
                value = loss(t["z"].to(kind))
            out[f"{name}-{dtype}-no_grad"] = value.double().numpy().tobytes(), b""
            for scale in (1.0, SCALE):
                z = t["z"].to(kind).clone().requires_grad_()
                value = loss(z * 1)  # logits of the graph, as in training
                (value * scale).backward()
                grad = z.grad
                assert grad is not None
                bits = grad.view(torch.int16) if grad.element_size() == 2 else grad
                key = f"{name}-{dtype}-{scale}"
                out[key] = value.detach().double().numpy().tobytes(), bits.numpy().tobytes()
    return out


def jax_outputs() -> dict[str, tuple[bytes, bytes]]:
    import jax
    import jax.numpy as jnp
    import libsdca.jax as J

    jax.config.update("jax_enable_x64", True)
    out = {}
    for dtype in DTYPES:
        raw = _inputs(np.random.default_rng(0))
        t = {key: jnp.asarray(a) for key, a in raw.items()}
        t["weight"] = t["weight"].astype(dtype if dtype == "float64" else "float32")
        t["teacher"] = t["teacher"].astype(dtype)
        t["probabilities"] = t["probabilities"].astype(dtype)
        t["kept"] = jnp.clip(t["labels"], 0)
        z = t["z"].astype(dtype)
        for name, loss in _losses(J, t).items():
            for scale in (1.0, SCALE):
                v, g = jax.jit(jax.value_and_grad(lambda z, f=loss, s=scale: f(z) * s))(z)
                out[f"{name}-{dtype}-{scale}"] = np.asarray(v).tobytes(), np.asarray(g).tobytes()
            v, g = jax.vmap(jax.value_and_grad(loss))(jnp.stack([z, z * 2]))
            out[f"{name}-{dtype}-vmap"] = np.asarray(v).tobytes(), np.asarray(g).tobytes()
        # vmap of the label smoothing and of the class weights: a kernel call per group.
        smoothed = jax.vmap(
            jax.value_and_grad(lambda z, e, y=t["kept"]: J.cross_entropy(z, y, label_smoothing=e))
        )
        v, g = jax.jit(smoothed)(jnp.stack([z, z]), jnp.array([0.0, 0.3]))
        out[f"ce_smoothing_vmap-{dtype}"] = np.asarray(v).tobytes(), np.asarray(g).tobytes()
        weighted = jax.vmap(
            jax.value_and_grad(lambda z, w, y=t["kept"]: J.cross_entropy(z, y, weight=w)),
            in_axes=(None, 0),
        )
        v, g = weighted(z, jnp.stack([t["weight"], t["weight"] * 2]))
        out[f"ce_weight_vmap-{dtype}"] = np.asarray(v).tobytes(), np.asarray(g).tobytes()
    return out


def explain(framework: str) -> tuple[str, str]:
    if framework == "torch":
        import libsdca.torch as L
        import torch

        return L.explain_backend(L.cross_entropy, torch.zeros(2, 3), torch.tensor([0, 1]))
    import jax.numpy as jnp
    import libsdca.jax as J

    return J.explain_backend(J.cross_entropy, jnp.zeros((2, 3)), jnp.array([0, 1]))


def run(framework: str, path: str, what: str = "outputs") -> None:
    """The child: the outputs of framework (unless what is "none"), its explain_backend and
    the warnings of libsdca, pickled into path."""
    outputs: dict[str, tuple[bytes, bytes]] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explained = explain(framework)
        if what != "none":
            outputs = torch_outputs() if framework == "torch" else jax_outputs()
    messages = [str(w.message) for w in caught if str(w.message).startswith("libsdca")]
    with open(path, "wb") as f:
        pickle.dump({"outputs": outputs, "explain": explained, "warnings": messages}, f)


if __name__ == "__main__":
    run(*sys.argv[1:])

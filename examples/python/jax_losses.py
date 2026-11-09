"""libsdca losses in JAX: jit, grad, vmap, named functions and distillation."""

import jax
import jax.numpy as jnp
import libsdca
import numpy as np
from libsdca.jax import (
    cross_entropy,
    distillation_loss,
    multilabel_hinge_loss,
    sdca_loss,
    topk_cross_entropy,
    topk_hinge_loss,
)

key_x, key_w, key_t = jax.random.split(jax.random.PRNGKey(0), 3)
x = jax.random.normal(key_x, (64, 10))
y = jnp.argmax(x @ jax.random.normal(key_w, (10, 5)), axis=1)  # labels 0..4


# A linear model trained by gradient descent on a jitted top-k loss.
def objective(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    return sdca_loss(x @ w, y, "topk_hinge_beta", k=2, gamma=0.5)


step = jax.jit(jax.value_and_grad(objective))
w = jnp.zeros((10, 5))
for i in range(101):
    value, grad = step(w, x, y)
    w = w - 0.5 * grad
    if i % 25 == 0:
        print(f"step {i:3d}  loss {float(value):.4f}")

# Same values as the NumPy loss_and_grad on the same scores.
scores = x @ w
values, _ = libsdca.loss_and_grad(
    np.asarray(scores), np.asarray(y), "topk_hinge_beta", k=2, gamma=0.5
)
np.testing.assert_allclose(float(objective(w, x, y)), values.mean(), rtol=1e-5)

# Named functions; entropy supports higher derivatives.
print("cross_entropy", float(cross_entropy(scores, y)))
print("topk_cross_entropy", float(topk_cross_entropy(scores, y, k=2)))
print("topk_hinge beta", float(topk_hinge_loss(scores, y, k=2, variant="beta")))
mask = jax.nn.one_hot(y, 5, dtype=bool).at[:, 0].set(True)
print("multilabel_hinge", float(multilabel_hinge_loss(scores, mask, gamma=0.5)))
hessian = jax.hessian(lambda z: cross_entropy(z, y[0]))(scores[0])
print("entropy Hessian", hessian.shape)

# vmap over a leading batch axis, reduction="none" per example.
per_example = jax.vmap(lambda z, t: sdca_loss(z, t, "topk_cross_entropy", k=2, reduction="none"))
print("vmap", per_example(scores.reshape(4, 16, 5), y.reshape(4, 16)).shape)

# Distillation: both arguments are logits; the teacher gets no gradient.
teacher = scores + 0.1 * jax.random.normal(key_t, scores.shape)
g = jax.grad(lambda s: distillation_loss(s, teacher, temperature=2.0))(scores)
zero = jax.grad(lambda s: distillation_loss(s, scores, 2.0))(scores)
assert float(jnp.abs(zero).max()) == 0.0  # exactly zero at equal logits
print("distillation grad norm", float(jnp.linalg.norm(g)))

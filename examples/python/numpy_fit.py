"""NumPy interface: fit, predict, loss_and_grad, warm start, coordinate_update and Solver."""

import libsdca
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 10))
y = (X @ rng.normal(size=(10, 4))).argmax(axis=1)  # labels 0..3

# Minimize 0.5*||W||^2 + C * sum_i loss(X[i] @ W, y[i]); no intercept.
model = libsdca.fit(X, y, "topk_cross_entropy", k=2, C=1.0, tol=1e-6)
print(model.status, model.epochs, model.history["relative_gap"][-1])
assert model.status == "solved"
assert model.weights is not None  # None only for precomputed kernels
print("weights", model.weights.shape, "dual", model.dual.shape)
print("train accuracy", model.score(X, y))
probabilities = libsdca.fit(X, y, "cross_entropy").predict_proba(X[:3])  # entropy objectives only
assert np.allclose(probabilities.sum(axis=1), 1)

# Warm start: a short run, then continue from its dual variables.
short = libsdca.fit(X, y, "topk_hinge_alpha", k=2, gamma=0.5, max_epoch=2, eval_epoch=1)
resumed = libsdca.fit(X, y, "topk_hinge_alpha", k=2, gamma=0.5, initial_dual=short.dual)
print("warm start:", short.status, "->", resumed.status, resumed.epochs, "more epochs")

# Losses and gradients of any scores (n, classes), per row.
scores = model.decision_function(X)  # X @ model.weights
values, gradients = libsdca.loss_and_grad(scores, y, "topk_hinge_beta", k=2, gamma=0.5)
print("mean smooth top-2 hinge", values.mean(), gradients.shape)
mask = np.zeros(scores.shape, dtype=bool)
mask[np.arange(len(y)), y] = True
mask[:, 0] = True  # label sets for the multilabel losses
values, _ = libsdca.loss_and_grad(scores, mask, "multilabel_hinge")

# One exact SDCA coordinate step: entropy blocks satisfy
# A/C = onehot(y) - softmax(scores + norm2 * A) from previous=0.
z, C, norm2 = scores[:5], 1.0, 2.0
A = libsdca.coordinate_update(z, y[:5], "cross_entropy", C=C, norm2=norm2)
u = z + norm2 * A
softmax = np.exp(u - u.max(axis=1, keepdims=True))
softmax /= softmax.sum(axis=1, keepdims=True)
assert np.allclose(A / C, np.eye(4)[y[:5]] - softmax, atol=1e-10)

# float32 storage solves in float64 by default; precision="float32" is faster.
fast = libsdca.fit(X.astype(np.float32), y, precision="float32")
assert fast.weights is not None
print("float32:", fast.weights.dtype, fast.status)

# A Solver pins one precision (and native build) for repeated calls.
solver = libsdca.Solver("float64")
print("solver", solver.version, solver.fit(X, y, "topk_hinge_beta", k=2).status)

"""NumPy interface: fit, predict, loss_and_grad, warm start and coordinate_update."""

import libsdca
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 10))
y = (X @ rng.normal(size=(10, 4))).argmax(axis=1)  # labels 0..3

# Minimize 0.5*||W||^2 + C * sum_i loss(X[i] @ W, y[i]); no intercept.
model = libsdca.fit(X, y, libsdca.TopKCrossEntropy(k=2), C=1.0, tol=1e-6)
print(model.status, model.epochs, model.history["relative_gap"][-1])
assert model.status == "solved"
assert model.weights is not None  # None only for libsdca.Kernel(K) input
print("weights", model.weights.shape, "dual", model.dual.shape)
print("train accuracy", model.score(X, y))
probabilities = libsdca.fit(X, y).predict_proba(X[:3])  # cross-entropy losses only
assert np.allclose(probabilities.sum(axis=1), 1)

# Warm start: a short run, then continue from its dual variables.
hinge = libsdca.TopKHingeAlpha(k=2, gamma=0.5)
short = libsdca.fit(X, y, hinge, max_epoch=2, eval_epoch=1)
resumed = libsdca.fit(X, y, hinge, initial_dual=short.dual)  # checked, W built from it
print("warm start:", short.status, "->", resumed.status, resumed.epochs, "more epochs")

# Losses and gradients of any scores (n, classes), per row.
scores = model.decision_function(X)  # X @ model.weights.T
values, gradients = libsdca.loss_and_grad(scores, y, libsdca.TopKHingeBeta(k=2, gamma=0.5))
print("mean smooth top-2 hinge", values.mean(), gradients.shape)
mask = np.zeros(scores.shape, dtype=bool)
mask[np.arange(len(y)), y] = True
mask[:, 0] = True  # label sets for the multilabel losses
values, _ = libsdca.loss_and_grad(scores, mask, libsdca.MultilabelHinge())
# Distillation from teacher logits, and training on target probabilities.
values, _ = libsdca.loss_and_grad(scores, 2 * scores, libsdca.Distillation(temperature=2.0))
teacher = np.exp(scores - scores.max(axis=1, keepdims=True))
teacher /= teacher.sum(axis=1, keepdims=True)
student = libsdca.fit(X, libsdca.Probabilities(teacher))
print("distilled:", student.status)

# One exact SDCA coordinate step: entropy blocks satisfy
# A/C = onehot(y) - softmax(scores + norm2 * A) from previous=0.
z, C, norm2 = scores[:5], 1.0, 2.0
A = libsdca.coordinate_update(z, y[:5], libsdca.CrossEntropy(), C=C, norm2=norm2)
u = z + norm2 * A
softmax = np.exp(u - u.max(axis=1, keepdims=True))
softmax /= softmax.sum(axis=1, keepdims=True)
assert np.allclose(A / C, np.eye(4)[y[:5]] - softmax, atol=1e-10)

# float32 storage solves in float32 by default; precision="float64" keeps double iterates.
fast = libsdca.fit(X.astype(np.float32), y)
assert fast.weights is not None
print("float32:", fast.weights.dtype, fast.status)
exact = libsdca.fit(X.astype(np.float32), y, precision="float64")
print("float32 storage, float64 work:", exact.weights.dtype if exact.weights is not None else None)

# A Gram matrix (kernel input) trains the duals; decision_function takes K(test, train).
kernel = libsdca.fit(libsdca.Kernel(X @ X.T), y)
print("kernel:", kernel.status, kernel.decision_function(X[:3] @ X.T).shape)

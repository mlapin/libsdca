# Using `libsdca` (Guide for AI Coding Assistants)

This file is a concise reference for AI coding assistants integrating `libsdca` into
user codebases (PyTorch, JAX, scikit-learn, NumPy, Rust, C++, C, MATLAB).

## 1. Installation

Install only the extras the user's project needs:

```bash
pip install libsdca              # NumPy solver & losses only (requires numpy>=1.23)
pip install 'libsdca[torch]'     # PyTorch loss layers (requires torch>=2.1, installs ninja)
pip install 'libsdca[jax]'       # JAX loss functions (requires jax>=0.4.35)
pip install 'libsdca[sklearn]'   # scikit-learn SDCAClassifier (requires scikit-learn>=1.6)
cargo add libsdca                # Rust crate (requires Rust >=1.74 and a C++11 compiler)
```

- **Prebuilt wheels** exist for Linux `x86_64` (`manylinux_2_28`) and macOS `arm64` (11+).
- **First-use C++/CUDA compilation:** PyTorch and JAX compile small C++ extensions on first
  call when a C++17/C++20 compiler with OpenMP and Python headers (`build-essential` and
  `python3-dev` on Ubuntu/Debian; `xcode-select --install` on macOS) are present. Without
  them, `libsdca` still works and emits a one-time fallback warning (set
  `SDCA_CPU_EXTENSION=off` or `SDCA_CUDA_LAUNCHER=off` to silence it).

## 2. Which API to Use

| User goal | Recommended import & call |
| --- | --- |
| Drop-in replacement for `torch.nn.CrossEntropyLoss` | `from libsdca import CrossEntropyLoss, cross_entropy` |
| LLM / large-vocabulary linear head + CE without materializing `(N, V)` logits | `from libsdca import LinearCrossEntropyLoss, linear_cross_entropy` |
| Top-$k$ classification in PyTorch (`k` in `[1, C - 1]`) | `from libsdca import TopKCrossEntropyLoss, TopKHingeLoss` |
| Multiclass or multilabel SVM / CE loss in PyTorch | `from libsdca import MulticlassHingeLoss, MultilabelHingeLoss, MultilabelCrossEntropyLoss` |
| Teacher-student logit distillation in PyTorch | `from libsdca import DistillationLoss, distillation_loss` |
| Loss functions in JAX (`jit`, `grad`, `vmap`) | `import libsdca.jax as sjax` (`sjax.cross_entropy`, `sjax.topk_cross_entropy`, `sjax.distillation_loss`, `sjax.sdca_loss`) |
| Train a linear or kernel SVM / softmax classifier (scikit-learn API) | `from libsdca import SDCAClassifier` |
| Train a linear or kernel classifier with duality-gap history & warm starts | `import libsdca; model = libsdca.fit(X, y, objective="cross_entropy", C=1.0, tol=1e-3)` |

## 3. Critical Rules & Pitfalls to Avoid

1. **Class axis on 3-D+ tensors in PyTorch (`dim=1` vs. `dim=-1`):**
   - For standard **2-D `(N, C)`** (and 1-D `(C,)`) logits, `dim=1` and `dim=-1` are the
     same axis, so all losses work identically without specifying `dim`.
   - For **3-D+ tensors**, `CrossEntropyLoss` and `libsdca.cross_entropy` follow
     `torch.nn.CrossEntropyLoss` with classes on **`dim=1`** (`(N, C, ...)`), whereas all
     other PyTorch losses (`TopKCrossEntropyLoss`, `TopKHingeLoss`, `MulticlassHingeLoss`,
     `DistillationLoss`, `sdca_loss`) and all JAX losses default to **`dim=-1`** (`(B, T, C)`).
     Pass `dim=1` explicitly when feeding `(N, C, H, W)` tensors to non-CE losses.
2. **Label smoothing and probability targets in PyTorch:**
   - `CrossEntropyLoss` only supports integer `int64` class indices and `label_smoothing=0.0`.
   - For label smoothing or soft probability targets $q$, pass log-probabilities as the
     teacher to `distillation_loss`:
     ```python
     q = (1 - eps) * F.one_hot(target, num_classes).to(logits.dtype) + eps / num_classes
     loss = distillation_loss(logits, q.log(), temperature=1.0)
     ```
3. **Distillation takes raw logits and applies no $T^2$ multiplier:**
   - `distillation_loss(student_logits, teacher_logits, temperature=T)` expects **unnormalized
     logits** for both arguments (never probabilities), detaches the teacher automatically,
     and returns gradient $(q - p) / T$. Multiply the loss by `T * T` yourself if a classic
     Hinton $T^2$-scaled objective is desired.
4. **Unmaterialized LLM head (`linear_cross_entropy`):**
   - Signature: `linear_cross_entropy(hidden, weight, target, bias=None, *, class_weight=None, ignore_index=-100, reduction="mean", chunk_bytes=None, recompute=False)`.
   - Shapes match `F.linear`: `hidden` is `(N, H)` or `(B, T, H)`, `weight` is `(V, H)`,
     `bias` is `(V,)`, and `target` is `(N,)` or `(B, T)` (`int64`).
   - Pass `recompute=True` when training with a `GradScaler` if subnormal gradient tails
     must be preserved after unscaling.
5. **In-place CUDA backward memory optimization:**
   - On CUDA, `CrossEntropyLoss`, $k=1$ hinges, and multilabel losses overwrite the `logits`
     tensor in place during `backward()` **only if no other Python variable references
     `logits`**. Write `loss = criterion(model(x), target)` (or `del logits` before
     `loss.backward()`) to save an `(N, C)` allocation.
   - Inside `torch.compile`, Inductor disables in-place logits overwrite; to compile the
     network while keeping in-place loss memory savings, wrap the loss with
     `torch.compiler.disable`.
6. **SDCA solver (`libsdca.fit` and `SDCAClassifier`):**
   - Minimizes $0.5\|W\|_2^2 + C \sum_{i=1}^n \ell(x_i^\top W, y_i)$ with **no implicit
     intercept** (append a constant column of ones to `X` if an intercept is needed).
   - To match an averaged objective $\frac{\lambda}{2}\|W\|_2^2 + \frac{1}{n}\sum_{i=1}^n \ell_i$,
     set `C = 1.0 / (lambda * n)`.
   - Standardize features (`StandardScaler`) before calling `fit`: coordinate step sizes
     scale with $1 / (C \|x_i\|_2^2)$.
   - For wide problems ($d \gg n$), precompute `K = X @ X.T` and pass `kernel=True`
     (`kernel="precomputed"` in `SDCAClassifier`) for up to `40×` faster solves.
   - Check `model.status == "solved"` to verify the duality-gap certificate `tol` was met.

## 4. Copy-Paste Recipes

### PyTorch: Cross-Entropy, Top-$k$, and Unmaterialized Linear Head

```python
import torch
from libsdca import CrossEntropyLoss, TopKCrossEntropyLoss, linear_cross_entropy

# 1. Drop-in CrossEntropyLoss (classes on dim=1, int64 targets)
criterion = CrossEntropyLoss(ignore_index=-100)
loss = criterion(model(x), target)
loss.backward()

# 2. Top-k cross-entropy (classes on dim=-1 by default; requires 1 <= k < num_classes)
topk_criterion = TopKCrossEntropyLoss(k=5)
loss = topk_criterion(model(x), target)
loss.backward()

# 3. LLM head without materializing (B * T, V) logits
hidden = backbone(tokens)  # (B, T, H), e.g. bfloat16 on CUDA
loss = linear_cross_entropy(hidden, lm_head.weight, labels, ignore_index=-100)
loss.backward()
```

### PyTorch & JAX: Knowledge Distillation

```python
from libsdca import DistillationLoss

distill = DistillationLoss(temperature=2.0)
loss = distill(student_logits, teacher_logits)  # exact 0 gradient when student == teacher
loss.backward()
```

```python
import jax
import libsdca.jax as sjax

loss_and_grad = jax.jit(jax.value_and_grad(
    lambda w, x, y: sjax.sdca_loss(x @ w, y, "topk_hinge_beta", k=3, gamma=0.5)
))
```

### Python: Certified SDCA Classifier (`scikit-learn` & `NumPy`)

```python
import libsdca
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# scikit-learn pipeline
pipe = make_pipeline(
    StandardScaler(),
    libsdca.SDCAClassifier("topk_hinge_beta", k=3, gamma=1.0, C=0.1, tol=1e-3),
)
pipe.fit(X_train, y_train)

# Functional API with duality-gap certificate
model = libsdca.fit(X_scaled, y_train, "multiclass_hinge", C=0.1, tol=1e-4)
assert model.status == "solved"
scores = model.decision_function(X_test)
```

## 5. Further Documentation

- Full loss formulas, options, and environment variables: [docs/API.md](docs/API.md)
- PyTorch guide: [python/TORCH.md](python/TORCH.md)
- JAX guide: [python/JAX.md](python/JAX.md)
- NumPy & scikit-learn guide: [python/README.md](python/README.md)
- Rust guide: [rust/README.md](rust/README.md)
- C++ / C / MATLAB guides: [docs/CPP.md](docs/CPP.md), [docs/C.md](docs/C.md), [docs/MATLAB.md](docs/MATLAB.md)

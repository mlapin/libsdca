# Python examples

Small, deterministic scripts, one per API family. They run on CPU and use
CUDA when it is available. Install the extras they need first:

```bash
pip install 'libsdca[torch,jax,sklearn]'
python examples/python/torch_drop_in_cross_entropy.py
```

| Script | Shows | Needs |
| --- | --- | --- |
| [torch_drop_in_cross_entropy.py](torch_drop_in_cross_entropy.py) | Replaces `torch.nn.CrossEntropyLoss` in an MLP training loop, checks the loss and gradient against PyTorch (`weight`, `ignore_index`, `(N, C, d1, d2)`), and switches to top-*k*. | torch |
| [torch_losses_tour.py](torch_losses_tour.py) | Runs every PyTorch loss as a module and as a function, forward and backward. | torch |
| [torch_mnist_lenet.py](torch_mnist_lenet.py) | Trains LeNet-5 on MNIST with `torch.nn.CrossEntropyLoss`, `libsdca.CrossEntropyLoss`, smoothed multiclass hinge and top-2 cross-entropy from one initialization. | torch |
| [torch_distillation.py](torch_distillation.py) | Teacher-student distillation, the zero gradient at equal logits, and proximal distillation tending to distillation as `alpha -> 0`. | torch |
| [numpy_fit.py](numpy_fit.py) | `fit`, `Model` prediction and probabilities, warm start, `loss_and_grad`, `coordinate_update` (checks its optimality condition) and `Solver`. | numpy |
| [sklearn_classifier.py](sklearn_classifier.py) | `SDCAClassifier` with string labels, in a `Pipeline` with `GridSearchCV`, with `predict_proba` and a precomputed RBF kernel. | scikit-learn |
| [jax_losses.py](jax_losses.py) | `sdca_loss` under `jit` and `value_and_grad`, agreement with NumPy, the named functions, `vmap`, an entropy Hessian and `distillation_loss`. | jax |

`python examples/python/torch_mnist_lenet.py --loss all --epochs 5` on the CPU of an Apple
M5 Pro (PyTorch 2.15.0, libsdca 0.3.0; batches of 128, SGD with momentum 0.9 and a
one-cycle schedule peaking at 0.05):

| Loss | Test top-1 | Test top-2 | Training seconds |
| --- | ---: | ---: | ---: |
| `torch.nn.CrossEntropyLoss` | 0.9930 | 0.9991 | 46.1 |
| `libsdca.CrossEntropyLoss` | 0.9928 | 0.9989 | 44.3 |
| `libsdca.MulticlassHingeLoss(gamma=1.0)` | 0.9927 | 0.9986 | 44.6 |
| `libsdca.TopKCrossEntropyLoss(k=2)` | 0.9910 | 0.9990 | 44.9 |

The two cross-entropies start from the same weights and batches and differ only in
rounding, which SGD amplifies: their epoch losses agree within 3% and their test
accuracies within half a point. The time is the network's; the
loss is a small part of it at 10 classes.

See the [loss definitions](../../docs/API.md#loss-definitions) for what each loss computes. C, C++
and Rust programs are in [examples/c](../c), [examples/cpp](../cpp) and
[rust/examples](../../rust/examples).

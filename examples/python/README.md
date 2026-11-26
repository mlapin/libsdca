# Python examples

Small, deterministic scripts, one per API family. They run on CPU and use
CUDA when it is available. Install the extras they need first:

```bash
pip install 'libsdca[torch,jax,sklearn]'
python examples/python/torch_drop_in_cross_entropy.py
```

| Script | Shows | Needs |
| --- | --- | --- |
| [torch_drop_in_cross_entropy.py](torch_drop_in_cross_entropy.py) | Replaces `torch.nn.CrossEntropyLoss` in an MLP training loop, checks the loss and gradient against PyTorch (`weight`, `ignore_index`, `label_smoothing`, probability targets, `(N, C, d1, d2)`), and switches to top-*k*. | torch |
| [torch_losses_tour.py](torch_losses_tour.py) | Runs every PyTorch loss except `linear_cross_entropy` as a module and as a function, forward and backward, with its `explain_backend` path, class weights and `torch.func` per-example gradients. | torch |
| [torch_mnist_lenet.py](torch_mnist_lenet.py) | Trains LeNet-5 on MNIST with `torch.nn.CrossEntropyLoss`, `libsdca.torch.CrossEntropyLoss`, smoothed multiclass hinge and top-2 cross-entropy from one initialization. | torch |
| [torch_distillation.py](torch_distillation.py) | Teacher-student distillation, the zero gradient at equal logits, and proximal distillation tending to distillation as `gamma -> 0`. | torch |
| [numpy_fit.py](numpy_fit.py) | `fit`, `Model` prediction and probabilities, warm start, `loss_and_grad` and `coordinate_update` (checks its optimality condition). | numpy |
| [sklearn_classifier.py](sklearn_classifier.py) | `SDCAClassifier` with string labels, in a `Pipeline` with `GridSearchCV`, with `predict_proba` and a precomputed RBF kernel. | scikit-learn |
| [jax_losses.py](jax_losses.py) | The top-2 hinge under `jit` and `value_and_grad`, agreement with NumPy, class weights, label smoothing and probability targets in `cross_entropy`, `vmap`, an entropy Hessian with `backend="composed"` and `distillation_loss`. | jax |

```bash
python examples/python/torch_mnist_lenet.py --loss all --epochs 5   # downloads MNIST
```

trains LeNet-5 with each loss from the same initialization and batches and prints test
top-1 and top-2 accuracy per loss.

See the [loss definitions](../../docs/API.md#loss-definitions) for what each loss computes. C, C++
and Rust programs are in [examples/c](../c), [examples/cpp](../cpp) and
[rust/examples](../../rust/examples).

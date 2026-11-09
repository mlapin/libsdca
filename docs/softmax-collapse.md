# Softmax collapse and cancellation

Barron, Gnanapragasam and Tschannen,
[*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642) (2026), identify
two floating-point failure modes in standard softmax cross-entropy. This page explains both,
shows how `libsdca` avoids them across all losses, and summarizes our reproduction of the
paper's experiments.

## Softmax collapse

For logits $z$ and target $y$, let $\tau = \sum_{j \neq y} \exp(z_j - z_y)$. Cross-entropy is
$\log(1 + \tau)$ and its target gradient is $p_y - 1 = -\tau / (1 + \tau)$. When $\tau$ falls
below $\frac{1}{2}\operatorname{ulp}(1)$ ($2^{-24} \approx 6\times 10^{-8}$ in `float32`,
$2^{-53} \approx 1.1\times 10^{-16}$ in `float64`), $1 + \tau$ rounds to $1$: both the loss
and target gradient collapse to exact zero while competitor gradients remain nonzero, breaking
the zero-sum gradient invariant. For two classes this occurs at margin $16.6$ (`float32`) or
$36.7$ (`float64`); with $C - 1$ equal competitors it shifts by $\log(C - 1)$.

`libsdca` evaluates the loss as $\operatorname{log1p}(\tau)$ and forms the target gradient
from the competitor sum (Eqs. 5–6 of the paper), remaining nonzero down to the output
dtype's subnormal limit (`104` in `float32`, `745` in `float64` for two classes):

```python
import torch
import libsdca

logits = torch.tensor([[0.0, -20.0]], requires_grad=True)
target = torch.tensor([0])
torch.nn.functional.cross_entropy(logits, target)    # tensor(0.)
loss = libsdca.cross_entropy(logits, target)         # tensor(2.0612e-09)
loss.backward()
logits.grad                                          # [[-2.0612e-09, 2.0612e-09]]
```

![Target gradient error against margin](figures/pytorch-softmax-collapse.png)

*Target-gradient relative error vs. prediction margin (1,000 classes, H100, 100-digit
reference). At error = 1, the gradient has collapsed to exact zero.*

The same formulation is used across every `libsdca` loss (top-$k$ cross-entropy, proximal
cross-entropy, multilabel cross-entropy, and the hinges) on CPU and CUDA in `float16`,
`bfloat16`, `float32` and `float64`.

## Softmax cancellation

The knowledge-distillation gradient is $\operatorname{softmax}(z_{\mathrm{S}} / T) - \operatorname{softmax}(z_{\mathrm{T}} / T)$.
As student logits $z_{\mathrm{S}}$ approach teacher logits $z_{\mathrm{T}}$, subtracting the
two rounded probability vectors loses significant digits: even at $z_{\mathrm{S}} = z_{\mathrm{T}}$,
PyTorch returns up to $4.6\times 10^{-9}$ in `float32` (`2.2\times 10^{-3}` in `bfloat16`)
instead of zero.

`libsdca.distillation_loss` evaluates each entry from the centered log-ratio $z_{\mathrm{S}} - z_{\mathrm{T}}$
using $\operatorname{expm1}$ (Eq. 9 of the paper), producing an **exact zero gradient** at
equal logits in every dtype:

```python
student = torch.randn(32, 10, requires_grad=True)
libsdca.distillation_loss(student, student.detach(), temperature=2.0).backward()
torch.count_nonzero(student.grad)                    # tensor(0)
```

![Soft-target gradient as the student approaches the teacher](figures/softmax-cancellation.png)

*Distillation gradient norm as student logits approach teacher logits ($T = 1$, modular
addition $a + b \bmod 53$ batch of $32 \times 53$ logits).*

## Relation to the paper

`libsdca` implements the paper's $\operatorname{log1p}$ and centered-$\operatorname{expm1}$
formulations inside its own fused CUDA, C++ ATen, JAX and C ABI kernels, independently of
external framework patches ([numerics.md](numerics.md)).

## Reproduction

We reproduced Sections 3, 6 and 7 of Barron et al. on an H100 (and verified on an A100):
modular addition grokking on `a + b mod 31` (Table 1, Fig. 5; 5 seeds $\times$ 10,000 epochs),
four grokking tasks (Fig. 6), and teacher-initialized distillation on `a + b mod 53` (Fig. 7).

Test accuracy (%) on `a + b mod 31` at Adam $\epsilon \in \{10^{-8}, 10^{-12}, 10^{-16}\}$
and final logit margin at $\epsilon = 10^{-16}$:

| dtype | Loss | Paper | Reproduction | Margin |
| --- | --- | --- | --- | ---: |
| `float32` | Fixed CE (paper) | 100.0 / 99.6 / 99.0 | 98.3 / 97.8 / 97.2 | 36.8 |
| `float32` | Standard CE | 94.3 / 96.4 / 96.5 | 92.5 / 93.4 / 92.8 | 12.9 |
| `float32` | `libsdca.CrossEntropyLoss` | — | 98.3 / 97.3 / 97.3 | 36.8 |
| `bfloat16` | `libsdca.CrossEntropyLoss` | — | 99.2 / 98.3 / 98.2 | 36.9 |
| `float32` | `libsdca.TopKHingeLoss(k=3, gamma=1, variant="alpha")` | — | 100.0 / 100.0 / 100.0 | 1.0 |

- **Collapse and fix reproduce:** Standard `float32` CE collapses on `73–88%` of training rows
  and stalls at margin `12.9`, whereas Fixed CE and `libsdca.CrossEntropyLoss` have zero
  collapsed rows and reach the theoretical equilibrium margins (`17.9 / 27.4 / 36.8`).
- **Cancellation reproduces:** Initialized at the teacher, standard `bfloat16` soft-target CE
  drifts to `17%` test accuracy (`KL -> 1.74`), whereas `DistillationLoss` and
  `ProximalDistillationLoss` have exact zero initial gradients and stay at `100%`.
- **Not reproduced from the paper:** StableMax (`77–81%` vs. paper's `95.6–97.4%`), standard
  CE in `bfloat16` (`80–83%` vs. `93–95%`), and `float64` standard CE margin at
  $\epsilon = 10^{-16}$ (`36.5` vs. `30.1`).

Full tables, training curves and the StableMax analysis are on the
[project page](https://mlapin.github.io/libsdca/).

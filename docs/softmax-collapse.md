# Softmax collapse and cancellation

Barron, Gnanapragasam and Tschannen,
[*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642) (2026), describe
two ways standard softmax cross-entropy loses its gradient in floating point. This page
explains both, shows what libsdca computes instead, and summarizes a reproduction of the
paper's experiments with the libsdca losses.

## Softmax collapse

Let `τ = Σ_{j≠y} exp(z_j − z_y)` for logits `z` and target `y`. Cross-entropy is
`log(1 + τ)` and its target gradient is `p_y − 1 = −τ / (1 + τ)`. When `τ` is below half
an ulp of 1 (`2⁻²⁴ ≈ 6e-8` in float32, `2⁻⁵³ ≈ 1.1e-16` in float64), `1 + τ` rounds to 1:
the loss and the target gradient become exactly zero, while the competitors' gradients
stay nonzero. The gradient of the row no longer sums to zero. With two classes this
happens once the target leads by about 16.6 (float32) or 36.7 (float64); with `C − 1`
competitors at the same gap, `τ = (C − 1) exp(−gap)` and the boundary moves up by
`log(C − 1)`.

libsdca computes the loss as `log1p(τ)` and the target gradient as minus the sum of the
competitor entries, as in Eqs. 5–6 of the paper. Both stay nonzero until the exact value
underflows the output dtype: with two classes near a margin of 104 in float32 and 745 in
float64.

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

*Relative error of the target gradient of a correct prediction (1000 classes, H100)
against 100-digit references. An error of 1 means the gradient is exactly zero.*

The same construction is used in every libsdca loss: top-k cross-entropy, the proximal
losses and the hinges form the target entry from the competitors, on CPU and CUDA and in
float16, bfloat16, float32 and float64.

## Softmax cancellation

The distillation gradient is `softmax(z_S / T) − softmax(z_T / T)`. When the student
approaches the teacher, the two probability vectors agree to many digits and their
difference is dominated by rounding: at equal logits PyTorch returns up to 4.6e-9 in
float32 instead of zero.

`libsdca.distillation_loss` forms each gradient entry from the centered log-ratio
`z_S − z_T`, using `expm1` where the two agree, as in Eq. 9 of the paper, with guards for
temperature and overflow. At equal logits the gradient is exactly zero in every dtype.

```python
student = torch.randn(32, 10, requires_grad=True)
libsdca.distillation_loss(student, student.detach(), temperature=2.0).backward()
torch.count_nonzero(student.grad)                    # tensor(0)
```

![Soft-target gradient as the student approaches the teacher](figures/softmax-cancellation.png)

## Relation to the paper

libsdca follows the paper's formulations (the `log1p` normalizer and the centered `expm1`
gradient) and applies them to all of its losses. Its kernels are its own; they do not use
the framework patches or packages that implement the paper's fix. The
[numerical notes](numerics.md) give the error bounds and how they are tested, including
the aggregate-tail cases where a result underflows.

## Reproduction

We reran Sections 3, 6 and 7 of the paper with standard cross-entropy, the paper's fixed
cross-entropy and the libsdca losses: grokking on `a + b mod 31` (Table 1, Fig. 5), four
grokking tasks (Fig. 6) and distillation from a trained teacher (Fig. 7). Table 1 uses
5 seeds × 10,000 epochs on an H100 with PyTorch 2.14.1.

Test accuracy (%) on `a + b mod 31` at Adam ε = 1e-8 / 1e-12 / 1e-16, and the logit
margin at ε = 1e-16:

| dtype | Loss | Paper | This run | Margin |
| --- | --- | --- | --- | ---: |
| float32 | Fixed CE (paper) | 100.0 / 99.6 / 99.0 | 98.3 / 97.8 / 97.2 | 36.8 |
| float32 | Standard CE | 94.3 / 96.4 / 96.5 | 92.5 / 93.4 / 92.8 | 12.9 |
| float32 | `libsdca.CrossEntropyLoss` | — | 98.3 / 97.3 / 97.3 | 36.8 |
| bfloat16 | `libsdca.CrossEntropyLoss` | — | 99.2 / 98.3 / 98.2 | 36.9 |
| float32 | `libsdca.TopKHingeLoss(k=3, gamma=1, variant="alpha")` | — | 100.0 / 100.0 / 100.0 | 1.0 |

Results:

- **Collapse reproduces.** Standard float32 cross-entropy collapses on 73–88% of the
  training rows at some point, and its margins stall below the paper's equilibrium. The
  paper's fix and every libsdca loss have no collapsed rows in any run.
- **The fix reproduces.** Fixed cross-entropy and `libsdca.CrossEntropyLoss` reach the
  paper's margins (17.9 / 27.4 / 36.8) to within 0.1 in all three precisions. Accuracy on
  seeds 0–4 is 0.6–2.2 points below the paper, mostly from one seed; on seeds 5–9 fixed
  cross-entropy reaches 99.9–100.0%.
- **Cancellation reproduces.** Starting from the teacher, standard soft-target
  cross-entropy in bfloat16 arithmetic drifts to 17% test accuracy. The paper's Eq. 9,
  `DistillationLoss` and `ProximalDistillationLoss` return an exactly zero gradient in
  bfloat16, float32 and float64 and stay at 100%.
- **Hinge losses** need only a margin of 1, independent of ε. The smooth top-k hinge α
  reaches 99.9–100.0%.
- **Not reproduced:** StableMax (77–81% against the paper's 95.6–97.4%), the accuracy of
  standard cross-entropy in bfloat16 arithmetic (80–83% against 93–95%), and the margin of
  float64 standard cross-entropy at ε = 1e-16 (36.5 against 30.1).

A rerun on an A100 with PyTorch 2.15 agrees with the H100 run: all 117 table cells and
189 of 190 training curves are within a 95% Welch interval. Both runs used 0.3.0
pre-releases. The CUDA kernels for short rows (up to 64 or 256 classes) changed after
them; the release kernels are covered by the CUDA numerics checks. The conclusions about
the paper do not depend on libsdca: standard and fixed cross-entropy and StableMax are
plain PyTorch.

The full report, with all tables, learning curves and the StableMax analysis, is on the
[project page](https://mlapin.github.io/libsdca/).

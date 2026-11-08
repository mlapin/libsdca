"""Logit distillation and proximal distillation with libsdca losses."""

import torch
from libsdca import (
    CrossEntropyLoss,
    DistillationLoss,
    ProximalDistillationLoss,
    distillation_loss,
    proximal_distillation_loss,
)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.randn(256, 16, device=device)
y = (X @ torch.randn(16, 8, device=device)).argmax(dim=1)


def mlp(width: int) -> torch.nn.Sequential:
    layers = [torch.nn.Linear(16, width), torch.nn.ReLU(), torch.nn.Linear(width, 8)]
    return torch.nn.Sequential(*layers).to(device)


# A teacher trained with cross-entropy, then a smaller student.
teacher, student = mlp(128), mlp(16)
opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
for _ in range(200):
    opt.zero_grad()
    CrossEntropyLoss()(teacher(X), y).backward()
    opt.step()
with torch.no_grad():
    teacher_logits = teacher(X)

# Both arguments are logits; the teacher is detached; no T**2 factor is applied.
distill, hard = DistillationLoss(temperature=4.0), CrossEntropyLoss()
opt = torch.optim.Adam(student.parameters(), lr=1e-2)
for step in range(201):
    opt.zero_grad()
    s = student(X)
    loss = 0.5 * hard(s, y) + 0.5 * 16.0 * distill(s, teacher_logits)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
agree = (student(X).argmax(1) == teacher_logits.argmax(1)).float().mean().item()
print(f"student/teacher agreement {agree:.3f}")

# The gradient is exactly zero at equal logits.
z = teacher_logits.clone().requires_grad_()
distillation_loss(z, teacher_logits, 4.0).backward()
assert torch.count_nonzero(z.grad) == 0

# Proximal distillation: a Moreau envelope that tends to distillation as alpha -> 0.
s = student(X).detach().double()
t = teacher_logits.double()
plain = distillation_loss(s, t, 4.0)
for alpha in (1.0, 1e-2, 1e-6):
    prox = proximal_distillation_loss(s, t, 4.0, alpha)
    print(f"alpha {alpha:7.0e}  proximal {prox.item():.6f}  plain {plain.item():.6f}")
torch.testing.assert_close(prox, plain, rtol=1e-4, atol=1e-6)
print("module form:", ProximalDistillationLoss(4.0, alpha=0.5)(s, t).item())

"""Swap torch.nn.CrossEntropyLoss for libsdca.CrossEntropyLoss in a training loop."""

import libsdca
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Synthetic 5-class problem: 512 points, 20 features.
X = torch.randn(512, 20, device=device)
y = (X @ torch.randn(20, 5, device=device)).argmax(dim=1)
net = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5)).to(
    device
)

# Before: criterion = torch.nn.CrossEntropyLoss()
criterion = libsdca.CrossEntropyLoss()

# Same value and gradient as PyTorch at initialization (float32 tolerance).
logits = net(X)
ours, ref = criterion(logits, y), torch.nn.CrossEntropyLoss()(logits, y)
g_ours = torch.autograd.grad(ours, logits, retain_graph=True)[0]
g_ref = torch.autograd.grad(ref, logits)[0]
torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(g_ours, g_ref, rtol=1e-4, atol=1e-7)

# Class weights, ignore_index and (N, C, d1, ...) inputs match too.
w = torch.tensor([1.0, 2.0, 0.5, 1.0, 3.0], device=device)
z = torch.randn(4, 5, 3, 2, device=device)
t = torch.randint(5, (4, 3, 2), device=device)
t[0, 0, 0] = -1
for reduction in ("none", "sum", "mean"):
    a = libsdca.CrossEntropyLoss(w, ignore_index=-1, reduction=reduction)(z, t)
    b = torch.nn.CrossEntropyLoss(w, ignore_index=-1, reduction=reduction)(z, t)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

# The training loop is unchanged.
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(101):
    optimizer.zero_grad()
    loss = criterion(net(X), y)
    loss.backward()
    optimizer.step()
    if step % 25 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
accuracy = (net(X).argmax(dim=1) == y).float().mean().item()
print(f"train accuracy {accuracy:.3f}")
assert accuracy > 0.9

# One-line switch to a top-k loss.
criterion = libsdca.TopKCrossEntropyLoss(k=2)
print("top-2 entropy:", criterion(net(X), y).item())

"""Train LeNet-5 on MNIST with libsdca's losses and compare them with torch.nn.CrossEntropyLoss.

    python examples/python/torch_mnist_lenet.py --loss libsdca-ce
    python examples/python/torch_mnist_lenet.py --loss all --epochs 5

The MNIST files are downloaded once into --data (about 11 MB) and read with NumPy, so the
script needs only torch and numpy. Every loss trains the same network from the same
initialization and batch order: torch-ce (torch.nn.CrossEntropyLoss), libsdca-ce
(libsdca.CrossEntropyLoss, a drop-in), libsdca-hinge (libsdca.MulticlassHingeLoss with
gamma = 1, the smoothed Crammer-Singer hinge) and libsdca-top2 (libsdca.TopKCrossEntropyLoss
with k = 2, scored on top-1 and top-2 accuracy).
"""

from __future__ import annotations

import argparse
import gzip
import time
import urllib.request
from pathlib import Path

import libsdca
import numpy as np
import torch
from torch import nn

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
LOSSES = {
    "torch-ce": lambda: nn.CrossEntropyLoss(),
    "libsdca-ce": lambda: libsdca.CrossEntropyLoss(),
    "libsdca-hinge": lambda: libsdca.MulticlassHingeLoss(gamma=1.0),
    "libsdca-top2": lambda: libsdca.TopKCrossEntropyLoss(k=2),
}


def idx(path: Path) -> np.ndarray:
    """An IDX file (big-endian header: zero, zero, type, rank; then the sizes) as an array."""
    raw = gzip.decompress(path.read_bytes())
    rank = raw[3]
    shape = np.frombuffer(raw, ">u4", rank, 4)
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * rank).reshape(shape)


def mnist(root: Path, split: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    root.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name in FILES[split]:
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + name, path)
        arrays.append(idx(path))
    images = torch.tensor(arrays[0], dtype=torch.float32).unsqueeze(1) / 255
    images = (images - 0.1307) / 0.3081
    return images.to(device), torch.tensor(arrays[1], dtype=torch.int64).to(device)


def lenet() -> nn.Module:
    """LeNet-5 for 28x28 inputs (padding 2 on the first convolution), ReLU and max pooling."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


@torch.no_grad()
def accuracy(net: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    top2 = torch.cat([net(x[i : i + 2000]).topk(2, dim=1).indices for i in range(0, len(x), 2000)])
    return (top2[:, 0] == y).float().mean().item(), (top2 == y[:, None]).any(
        1
    ).float().mean().item()


def train(name: str, args: argparse.Namespace, data: dict, device: str) -> dict:
    torch.manual_seed(args.seed)
    net = lenet().to(device)
    criterion = LOSSES[name]()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, epochs=args.epochs, steps_per_epoch=-(-60000 // args.batch)
    )
    x, y = data["train"]
    order = torch.Generator().manual_seed(args.seed)
    seconds, top1, top2 = 0.0, 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        permutation = torch.randperm(len(x), generator=order).to(device)
        start, total = time.perf_counter(), 0.0
        for i in range(0, len(x), args.batch):
            batch = permutation[i : i + args.batch]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item() * len(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        net.eval()
        top1, top2 = accuracy(net, *data["test"])
        print(
            f"{name:14s} epoch {epoch}  loss {total / len(x):.4f}  "
            f"test top-1 {top1:.4f}  top-2 {top2:.4f}  {seconds:.1f} s"
        )
    return {"loss": name, "top1": top1, "top2": top2, "seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=[*LOSSES, "all"], default="libsdca-ce")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache" / "mnist")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    data = {split: mnist(args.data, split, args.device) for split in FILES}
    results = [
        train(n, args, data, args.device) for n in (LOSSES if args.loss == "all" else [args.loss])
    ]
    print("\n| Loss | Test top-1 | Test top-2 | Training seconds |\n| --- | ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['loss']} | {r['top1']:.4f} | {r['top2']:.4f} | {r['seconds']:.1f} |")
    assert all(r["top1"] > 0.97 for r in results), "LeNet-5 reaches over 97% within 3 epochs"


if __name__ == "__main__":
    main()

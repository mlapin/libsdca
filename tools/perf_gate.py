"""CUDA runtime and memory gate: libsdca losses against torch.compile of the same objectives.

    PYTHONPATH=python python tools/perf_gate.py [--output gate.json] [--quick]

Forward and backward on wide rows, timed with CUDA events in alternating blocks. Each case
first checks that both implementations compute the same loss. Exit 1 when a speedup falls
below its floor or libsdca needs more peak memory than the compiled reference. The floors
sit below the recorded speedups (docs/TORCH.md, Benchmarks), so a failure means a
regression, not timing noise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable

import libsdca.torch
import torch
import torch.nn.functional as F

K, T = 5, 2.0
# Logits ~ N(0, 9) and teachers within N(0, 1/4) of them, as the benchmarks. At 128,256
# classes distillation is slower on rows whose (student - teacher) / T spans more than
# +-2 around its mean (bf16 below torch.compile at a teacher noise of 1).
SCALE, TEACHER_NOISE = 3.0, 0.5
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def ref_hinge(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    h = 1 + z - z.gather(1, y[:, None])
    return h.scatter(1, y[:, None], -torch.inf).amax(1).clamp_min(0).mean()


def ref_topk_cross_entropy(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed form: the m largest competitors capped at 1/k, the rest by softmax."""
    yy = y[:, None]
    v, i = z.topk(K + 1, dim=1)
    x = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
    t, order = v.to(x.dtype).masked_fill(i == yy, -math.inf).sort(dim=1, descending=True)
    t, chosen = t[:, :K], i.gather(1, order[:, :K])
    last = t[:, -1:]
    other = torch.ones_like(x, dtype=torch.bool).scatter(1, chosen, False).scatter(1, yy, False)
    tail = last + torch.log(torch.where(other, torch.exp(x - last), 0).sum(1, keepdim=True))
    suffix = torch.logcumsumexp(torch.cat([t, tail], 1).flip(1), 1).flip(1)[:, :K]
    m = torch.arange(K, device=x.device, dtype=x.dtype)
    free = K - m
    capped = (t - suffix > -torch.log(free)) & (m < K - 1)
    count = torch.cumprod(capped, 1).sum(1, keepdim=True)
    head = torch.cumsum(t, 1) - t
    value = head / K + (m / K) * math.log(K) + (free / K) * (suffix - torch.log(free / K))
    odds = value.gather(1, count) - x.gather(1, yy)
    return torch.logaddexp(odds, torch.zeros((), dtype=x.dtype, device=x.device)).mean()


def ref_distillation(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(s / T, F.softmax(t.detach() / T, -1))


# name: (libsdca, reference, target kind, speedup floor)
CASES: dict[str, tuple[Loss, Loss, str, float]] = {
    "cross_entropy": (libsdca.torch.cross_entropy, F.cross_entropy, "labels", 0.95),
    "topk_cross_entropy": (
        lambda z, y: libsdca.torch.topk_cross_entropy(z, y, k=K),
        ref_topk_cross_entropy,
        "labels",
        3.5,
    ),
    "multiclass_hinge": (libsdca.torch.multiclass_hinge_loss, ref_hinge, "labels", 3.0),
    "distillation": (
        lambda s, t: libsdca.torch.distillation_loss(s, t, temperature=T),
        ref_distillation,
        "teacher",
        1.1,
    ),
}
SHAPES = [(4096, 32000), (1024, 128256)]


def step(fn: Loss, z: torch.Tensor, target: torch.Tensor) -> None:
    torch.autograd.grad(fn(z, target), z)


def peak_bytes(fn: Loss, z: torch.Tensor, target: torch.Tensor) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step(fn, z, target)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def timings(fns: dict[str, Loss], z: torch.Tensor, target: torch.Tensor, blocks: int, calls: int):
    for fn in fns.values():
        for _ in range(5):
            step(fn, z, target)
    samples: dict[str, list[float]] = {name: [] for name in fns}
    names = list(fns)
    for block in range(blocks):
        for name in names if block % 2 == 0 else names[::-1]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls):
                step(fns[name], z, target)
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / calls)
    return samples


def run_case(name: str, n: int, c: int, dtype: torch.dtype, blocks: int, calls: int) -> dict:
    lib, ref, kind, floor = CASES[name]
    gen = torch.Generator(device="cuda").manual_seed(0)
    scores = SCALE * torch.randn(n, c, device="cuda", generator=gen)
    z = scores.to(dtype).requires_grad_()
    if kind == "labels":
        target = torch.randint(0, c, (n,), device="cuda", generator=gen)
    else:
        noise = TEACHER_NOISE * torch.randn(n, c, device="cuda", generator=gen)
        target = (z.detach().float() + noise).to(dtype)
    compiled = torch.compile(ref, dynamic=False)
    a, b = float(lib(z, target)), float(compiled(z, target))
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise SystemExit(f"{name} {n}x{c} {dtype}: libsdca {a} != reference {b}")
    samples = timings({"libsdca": lib, "compile": compiled}, z, target, blocks, calls)
    lib_ms, ref_ms = (statistics.median(samples[k]) for k in ("libsdca", "compile"))
    memory = {"libsdca": peak_bytes(lib, z, target), "compile": peak_bytes(compiled, z, target)}
    speedup = ref_ms / lib_ms
    passed = speedup >= floor and memory["libsdca"] <= 1.05 * memory["compile"] + 2**20
    return dict(
        loss=name,
        shape=[n, c],
        dtype=str(dtype).removeprefix("torch."),
        libsdca_ms=lib_ms,
        compile_ms=ref_ms,
        speedup=speedup,
        floor=floor,
        peak_bytes=memory,
        samples_ms=samples,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA runtime and memory gate")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true", help="one shape, fewer blocks")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("perf_gate.py needs a CUDA device")
    blocks, calls = (6, 5) if args.quick else (15, 10)
    shapes = SHAPES[:1] if args.quick else SHAPES
    rows = []
    for name in CASES:
        for n, c in shapes:
            for dtype in (torch.bfloat16, torch.float32):
                row = run_case(name, n, c, dtype, blocks, calls)
                rows.append(row)
                print(
                    f"{'ok  ' if row['passed'] else 'FAIL'} {name:20s} {n}x{c} {row['dtype']:8s} "
                    f"{row['speedup']:5.2f}x (floor {row['floor']}), peak "
                    f"{row['peak_bytes']['libsdca'] / 2**20:.0f} vs "
                    f"{row['peak_bytes']['compile'] / 2**20:.0f} MiB",
                    flush=True,
                )
    meta = dict(
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        libsdca=getattr(libsdca, "__version__", None),
    )
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(dict(metadata=meta, rows=rows), stream, indent=1)
    sys.exit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()

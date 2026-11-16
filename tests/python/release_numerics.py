"""Bounded, independent entropy and distillation checks for a release candidate.

    PYTHONPATH=python python tests/python/release_numerics.py --device cpu --output DIR
    PYTHONPATH=python python tests/python/release_numerics.py --device cuda --output DIR

Use --dtype to split GPU runs; --jobs N checks the dtypes in N processes (the same rows
in the same order). Each result records represented inputs, backend,
precision and versions. Exit 1 means a libsdca check failed; PyTorch cancellation
is reported as a reference observation, never required to fail in future versions.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import platform
import subprocess
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mpmath as mp
import torch
import torch.nn.functional as F
from libsdca.torch import (
    distillation_loss,
    proximal_distillation_loss,
)
from named_losses import explain_route, torch_loss

DTYPES = ("float16", "bfloat16", "float32", "float64")
WIDTHS = (2, 3, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097, 32767, 32768, 32769)


def check_number(
    actual: float, expected: Any, dtype: torch.dtype, summed: float = 0.0, floor: float = 0.0
) -> tuple[bool, float | None]:
    """64 output eps relative (plus summed) plus one subnormal quantum (plus floor); no
    unit-scale floor."""
    info = torch.finfo(dtype)
    quantum = mp.mpf(info.tiny) * info.eps + floor
    if abs(expected) > info.max:
        return math.isinf(actual) and (actual > 0) == (expected > 0), None
    if not math.isfinite(actual):
        return False, None
    error = abs(mp.mpf(actual) - expected)
    budget = (64 * info.eps + summed) * abs(expected) + quantum
    ratio = float(error / budget) if budget else float(error)
    return error <= budget, ratio


def check_entries(
    name: str, entries: torch.Tensor, expected: Any, summed: float = 0.0, floor: float = 0.0
) -> list[tuple[str, float, Any, bool, float | None]]:
    """check_number on every distinct value of entries that share one exact reference."""
    values = torch.unique(entries.detach().double().cpu()).tolist()
    return [
        (name, value, expected, *check_number(value, expected, entries.dtype, summed, floor))
        for value in values
    ]


def reference(first: float, other: float, classes: int, label: int) -> tuple[Any, Any, Any, Any]:
    """Loss, target gradient, gradient of entry 0 when it is not the target (else None) and
    gradient of each other competitor, of the row [first, other, other, ...]."""
    gap = mp.mpf(first) - mp.mpf(other)
    if gap >= 0:
        e = mp.exp(-gap)
        tail = (classes - 1) * e
        loss = mp.log1p(tail) + (gap if label else 0)
        first_p, other_p = 1 / (1 + tail), e / (1 + tail)
    else:
        e = mp.exp(gap)
        den = classes - 1 + e
        loss = mp.log(den) - (gap if not label else 0)
        first_p, other_p = e / den, 1 / den
    target = -(classes - 1) * other_p if not label else -(first_p + (classes - 2) * other_p)
    return loss, target, first_p if label else None, other_p


def topk_reference(first: float, other: float, classes: int, label: int, k: int) -> tuple:
    """Top-k entropy (k < classes) as reference() returns cross-entropy: from the active set
    of the competitors, the k largest of them one by one and the rest by equal score. Tied
    competitors have equal gradients, so one share per group is the share of each."""
    if label == 0:
        groups = [(mp.mpf(other) - mp.mpf(first), classes - 1)]
    else:
        groups = [(mp.mpf(first) - mp.mpf(other), 1), (mp.mpf(0), classes - 2)]
    entries: list[tuple[Any, int, int]] = []  # (score - label score, count, group)
    for group, (v, count) in sorted(enumerate(groups), key=lambda g: -g[1][0]):
        single = min(count, k - len(entries)) if len(entries) < k else 0
        entries += [(v, 1, group)] * single
        if count > single:
            entries.append((v, count - single, group))
    capped, top, lse = 0, mp.mpf(0), mp.mpf(0)
    for capped in range(k):
        top = entries[capped][0]
        lse = mp.log(mp.fsum(n * mp.exp(v - top) for v, n, _ in entries[capped:]))
        if lse >= mp.log(k - capped):
            break
    free = k - capped
    odds = (mp.fsum(v for v, _, _ in entries[:capped]) + free * (top + lse - mp.log(free))) / k
    odds += mp.log(k)
    mass = 1 / (1 + mp.exp(-odds))
    shares = [
        mass / k if i < capped else mass * free / k * mp.exp(v - top - lse)
        for i, (v, _, _) in enumerate(entries)
    ]
    share = {group: s for s, (_, _, group) in reversed(list(zip(shares, entries, strict=True)))}
    loss = mp.log1p(mp.exp(odds))
    if label == 0:
        return loss, -mass, None, share[0]
    return loss, -mass, share[0], share.get(1)


# Gaps near and in the output's subnormal range, where entries built from sums of
# exponentials that underflow one by one lose bits or vanish.
SUBNORMAL_GAPS = {
    torch.float16: (87, 95, 103, 104, 106, 108),
    torch.bfloat16: (87, 95, 103, 104, 106, 108),
    torch.float32: (87, 95, 103, 104, 106, 108),
    torch.float64: (708, 715, 725, 735, 745),
}


def entropy_rows(args: argparse.Namespace, dtype: torch.dtype) -> Iterator[dict[str, Any]]:
    """cross_entropy and top-k entropy (k = 5, 64, 65 below the classes) of two-group rows.

    Includes CUDA fused and tensor top-k paths and subnormal target masses. Opposite-sign
    extreme finite limits remain outside this top-k sweep.
    Float32-work CPU reductions retain their established C-eps and subnormal budgets.
    """
    info = torch.finfo(dtype)
    gaps = (
        (0, 1, 20, 40, 100, 110, 740, 750) if dtype == torch.float64 else (0, 1, 20, 40, 100, 110)
    )
    gaps += SUBNORMAL_GAPS[dtype]
    widths = WIDTHS if not args.quick else (2, 65, 4097)
    for classes in widths:
        # The offsets straddle the point where input rounding destroys a unit gap.
        patterns = [(0, -gap) for gap in gaps]
        patterns += [(offset, offset - 1) for offset in (1 / info.eps, 2 / info.eps)]
        patterns += [(info.max, -info.max), (-info.max, info.max)]
        losses = [("cross_entropy", 1)] + [("topk_cross_entropy", k) for k in (5, 64, 65)]
        for (first, other), (kind, k) in itertools.product(patterns, losses):
            topk = k > 1
            if k >= classes or (topk and abs(first) == info.max):
                continue
            x = torch.full((2, classes), other, dtype=dtype, device=args.device)
            x[:, 0] = first
            represented = x[0, :2].double().cpu().tolist()
            labels = torch.tensor([0, 1], device=args.device)
            precisions = (None,) if dtype == torch.float64 else ("float32", "float64")
            implementations = [("torch", p) for p in precisions]
            if args.device == "cpu" and dtype in (torch.float32, torch.float64):
                implementations += [("native", p) for p in precisions]
            if args.device == "cpu" and os.environ.get("SDCA_CPU_EXTENSION") in ("on", "1"):
                # backend="native" bypasses the extension; "auto" takes it.
                implementations += [("auto", p) for p in precisions]
            if args.device == "cuda":
                implementations += [("auto", p) for p in precisions]
            for backend, precision in implementations:
                work = torch.finfo(torch.float32 if precision == "float32" else torch.float64)
                summed = classes * work.eps if topk and precision == "float32" else 0.0
                native32 = topk and backend == "native" and precision == "float32"
                floor = classes * work.tiny * work.eps / 2 if native32 else 0.0
                z = x.detach().requires_grad_()
                row: dict[str, Any] = dict(
                    kind=kind if k == 1 else f"{kind} k={k}",
                    dtype=str(dtype),
                    classes=classes,
                    represented=represented,
                    backend=backend,
                    precision=precision,
                )
                try:
                    loss = torch_loss(
                        z, labels, kind, k=k, reduction="none", backend=backend, precision=precision
                    )
                    (gradient,) = torch.autograd.grad(loss.sum(), z)
                    passed, worst, failures = True, 0.0, []
                    for i, label in enumerate((0, 1)):
                        value, target, first_entry, others = (
                            reference(represented[0], represented[1], classes, label)
                            if k == 1
                            else topk_reference(represented[0], represented[1], classes, label, k)
                        )
                        g = gradient[i]
                        groups = [
                            ("loss", loss[i : i + 1], value),
                            ("target", g[label : label + 1], target),
                        ]
                        if label:
                            groups.append(("first", g[:1], first_entry))
                        if classes > 1 + label:
                            groups.append(("others", g[1 + label :], others))
                        for name, entries, ref in groups:
                            for entry, actual, _, ok, ratio in check_entries(
                                name, entries, ref, summed, floor
                            ):
                                passed &= ok
                                worst = max(worst, ratio or 0)
                                if not ok:
                                    failures.append(
                                        dict(
                                            label=label,
                                            entry=entry,
                                            actual=actual,
                                            expected=mp.nstr(ref, 20),
                                            ratio=ratio,
                                        )
                                    )
                    row.update(passed=bool(passed), tolerance_ratio=worst)
                    if failures:
                        row["failures"] = failures
                except Exception as exc:
                    row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                yield row


def two_group(first: Any, other: Any, classes: int) -> tuple[Any, Any]:
    """Probabilities of the first entry and of each of the classes - 1 others."""
    e = mp.exp(mp.mpf(other) - mp.mpf(first))
    return 1 / (1 + (classes - 1) * e), e / (1 + (classes - 1) * e)


@functools.cache
def distillation_reference(
    student: tuple[float, ...],
    teacher: tuple[float, ...],
    classes: int,
    temperature: float,
    alpha: float | None,
    digits: int,
) -> tuple[Any, Any, Any]:
    """Loss and first/other gradients of (proximal) distillation on two-group rows, at the
    working digits (memoized: the represented rows repeat across dtypes).

    The prox solves v = (s_1 - s_2) / T - alpha k (P(v) - Q), k = C / (C - 1), by bisection.
    """
    q, q_other = two_group(
        mp.mpf(teacher[0]) / temperature, mp.mpf(teacher[1]) / temperature, classes
    )
    d = (mp.mpf(student[0]) - mp.mpf(student[1])) / temperature
    k = mp.mpf(classes) / (classes - 1)
    if alpha:
        lo, hi = d - alpha * k, d + alpha * k
        for _ in range(400):
            v = (lo + hi) / 2
            if v - d + alpha * k * (two_group(v, 0, classes)[0] - q) > 0:
                hi = v
            else:
                lo = v
        d = (lo + hi) / 2
    p, p_other = two_group(d, 0, classes)
    loss = -(q * mp.log(p) + (classes - 1) * q_other * mp.log(p_other))
    if alpha:
        loss += alpha * k * (p - q) ** 2 / 2
    return loss, (p - q) / temperature, (p_other - q_other) / temperature


def distillation_rows(args: argparse.Namespace, dtype: torch.dtype) -> Iterator[dict[str, Any]]:
    """distillation_loss and proximal_distillation_loss against 100-digit references.

    Rows sum C terms; on CUDA the budget adds C working epsilons (recursive summation in the
    tensor operations without Triton). The CPU extension and native library compensate them.
    """
    pairs = ((0, 0), (5, 5), (1, 1 + 1 / 64), (1, 0), (0, 1), (20, 1), (1, 20), (40, 0), (0, 40))
    widths = WIDTHS if not args.quick else (2, 65, 4097)
    settings = [("distillation", t, None) for t in (0.5, 2.0)]
    settings += [("proximal_distillation", 2.0, a) for a in (0.01, 1.0, 100.0)]
    precisions = (None,) if dtype == torch.float64 else ("float32", "float64")
    for classes in widths:
        for student_gap, teacher_gap in pairs:
            s = torch.full((1, classes), -student_gap, dtype=dtype, device=args.device)
            t = torch.full((1, classes), -teacher_gap, dtype=dtype, device=args.device)
            s[0, 0] = t[0, 0] = 0
            student, teacher = s[0, :2].double().tolist(), t[0, :2].double().tolist()
            for kind, temperature, alpha in settings:
                expected = distillation_reference(
                    tuple(student), tuple(teacher), classes, temperature, alpha, mp.mp.dps
                )
                for precision in precisions:
                    work = torch.float32 if precision == "float32" else torch.float64
                    summed = classes * torch.finfo(work).eps if args.device == "cuda" else 0.0
                    row: dict[str, Any] = dict(
                        kind=kind,
                        dtype=str(dtype),
                        classes=classes,
                        student=student,
                        teacher=teacher,
                        temperature=temperature,
                        alpha=alpha,
                        precision=precision,
                    )
                    try:
                        z = s.detach().requires_grad_()
                        if alpha:
                            loss = proximal_distillation_loss(
                                z,
                                t,
                                temperature=temperature,
                                alpha=alpha,
                                reduction="none",
                                precision=precision,
                            )
                        else:
                            loss = distillation_loss(
                                z, t, temperature=temperature, reduction="none", precision=precision
                            )
                        (gradient,) = torch.autograd.grad(loss.sum(), z)
                        groups = [
                            ("loss", loss[:1], expected[0]),
                            ("first", gradient[0, :1], expected[1]),
                            ("others", gradient[0, 1:], expected[2]),
                        ]
                        checked = [
                            (check, entries.dtype)
                            for name, entries, e in groups
                            for check in check_entries(name, entries, e, summed)
                        ]
                        row.update(
                            passed=all(ok for (_, _, _, ok, _), _ in checked),
                            tolerance_ratio=max(r or 0 for (_, _, _, _, r), _ in checked),
                            # Largest relative error in output epsilons (nonzero references).
                            error_eps=max(
                                (
                                    float(abs(mp.mpf(a) - e) / abs(e)) / torch.finfo(dtype).eps
                                    for (_, a, e, _, _), dtype in checked
                                    if e and math.isfinite(a)
                                ),
                                default=0.0,
                            ),
                        )
                        if not row["passed"]:
                            row["failures"] = [
                                dict(entry=name, actual=a, expected=mp.nstr(e, 20), ratio=r)
                                for (name, a, e, ok, r), _ in checked
                                if not ok
                            ]
                    except Exception as exc:
                        row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                    yield row


def barron_rows(args: argparse.Namespace, dtype: torch.dtype) -> Iterator[dict[str, Any]]:
    gap = 10 if dtype == torch.float16 else 40 if dtype == torch.float64 else 20
    for implementation in ("pytorch", "libsdca"):
        x = torch.tensor([[0.0, -gap]], dtype=dtype, device=args.device, requires_grad=True)
        y = torch.tensor([0], device=args.device)
        loss = (
            F.cross_entropy(x, y)
            if implementation == "pytorch"
            else torch_loss(x, y, backend="torch")
        )
        (gradient,) = torch.autograd.grad(loss, x)
        yield dict(
            kind="collapse",
            dtype=str(dtype),
            implementation=implementation,
            gap=gap,
            loss=loss.item(),
            target_gradient=gradient[0, 0].item(),
            competitor_gradient=gradient[0, 1].item(),
            passed=None
            if implementation == "pytorch"
            else loss.item() > 0 and gradient[0, 0].item() < 0,
        )
        source = torch.linspace(-2, 2, 65, dtype=dtype, device=args.device).reshape(1, -1)
        x = source.clone().requires_grad_()
        probability = source.softmax(-1)
        loss = (
            F.cross_entropy(x, probability)
            if implementation == "pytorch"
            else distillation_loss(x, source)
        )
        (gradient,) = torch.autograd.grad(loss, x)
        maximum = gradient.abs().max().item()
        row: dict[str, Any] = dict(
            kind="cancellation",
            dtype=str(dtype),
            implementation=implementation,
            maximum_gradient=maximum,
            passed=None if implementation == "pytorch" else maximum == 0,
        )
        if implementation == "pytorch":
            q = list(map(mp.mpf, probability[0].double().cpu().tolist()))
            mass = mp.fsum(q)
            z = list(map(mp.mpf, source[0].double().cpu().tolist()))
            e = [mp.exp(v - max(z)) for v in z]
            den = mp.fsum(e)
            expected = [mass * p / den - target for p, target in zip(e, q, strict=False)]
            row.update(
                operator="cross_entropy(student, softmax(teacher_logits))",
                represented_target_mass=mp.nstr(mass, 25),
                exact_probability_target_gradient_max=mp.nstr(max(map(abs, expected)), 25),
                comparison="composition observation; libsdca receives logits, PyTorch rounded probabilities",
            )
        else:
            row["operator"] = "distillation_loss(student, teacher_logits)"
        yield row


def available_cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def multilabel_rows(args: argparse.Namespace, dtype: torch.dtype) -> Iterator[dict[str, Any]]:
    """Boolean-mask entropy of well-fit rows (CPU: native, and auto with SDCA_CPU_EXTENSION;
    CUDA: the tensor and the kernel backends): two positives at 0 first, C - 2 negatives
    at -gap, whose C - 2 equal small terms beside a 1 lose digits in plain sums. Exact:
    den = 2 + (C - 2) exp(-gap), loss log(den), positive gradients 1 / den - 1 / 2, negative
    exp(-gap) / den (normal exponentials: gaps to 700 in float64). Each row records the route
    that explain_backend reports; auto must take the kernels (CUDA) or the extension (CPU)."""
    gaps = (0, 1, 20, 40, 70, 100, 110)
    gaps += (700,) if dtype == torch.float64 else SUBNORMAL_GAPS[dtype]
    widths = WIDTHS if not args.quick else (2, 65, 4097)
    backends = ["native"] if args.device == "cpu" else ["torch", "auto"]
    if args.device == "cpu" and os.environ.get("SDCA_CPU_EXTENSION") in ("on", "1"):
        backends.append("auto")
    precisions = (None,) if dtype == torch.float64 else (None, "float64")
    for classes in widths:
        for gap in gaps:
            x = torch.full((1, classes), -gap, dtype=dtype, device=args.device)
            x[0, :2] = 0
            mask = torch.zeros(1, classes, dtype=torch.bool, device=args.device)
            mask[0, :2] = True
            e = mp.exp(mp.mpf(x[0, -1].item()))
            den = 2 + (classes - 2) * e
            # The positives' gradient 1 / den - 1 / 2, without its cancellation.
            loss_ref, positive, negative = mp.log(den), -(classes - 2) * e / (2 * den), e / den
            for backend in backends:
                for precision in precisions:
                    route = explain_route(
                        x,
                        mask,
                        "multilabel_cross_entropy",
                        backend=backend,
                        reduction="none",
                        precision=precision,
                    )
                    if backend == "auto":
                        kernels = ("cuda-launcher", "triton") if x.is_cuda else ("cpu-kernel",)
                        if route[0] not in kernels:
                            raise SystemExit(f"auto takes {route[0]}, not a kernel: {route[1]}")
                    row: dict[str, Any] = dict(
                        kind="multilabel_cross_entropy",
                        dtype=str(dtype),
                        classes=classes,
                        gap=gap,
                        backend=backend,
                        route=route[0],
                        precision=precision,
                    )
                    try:
                        z = x.detach().requires_grad_()
                        loss = torch_loss(
                            z,
                            mask,
                            "multilabel_cross_entropy",
                            reduction="none",
                            backend=backend,
                            precision=precision,
                        )
                        (gradient,) = torch.autograd.grad(loss.sum(), z)
                        groups = [
                            ("loss", loss[:1], loss_ref),
                            ("positives", gradient[0, :2], positive),
                        ]
                        if classes > 2:
                            groups.append(("negatives", gradient[0, 2:], negative))
                        checked = [
                            check
                            for name, entries, r in groups
                            for check in check_entries(name, entries, r)
                        ]
                        row.update(
                            passed=all(ok for _, _, _, ok, _ in checked),
                            tolerance_ratio=max(q or 0 for _, _, _, _, q in checked),
                        )
                        if not row["passed"]:
                            row["failures"] = [
                                dict(entry=name, actual=a, expected=mp.nstr(r, 20), ratio=q)
                                for name, a, r, ok, q in checked
                                if not ok
                            ]
                    except Exception as exc:
                        row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                    yield row


def dtype_rows(args: argparse.Namespace, name: str) -> list[dict[str, Any]]:
    """Every row of one dtype, on one thread."""
    torch.set_num_threads(1)
    dtype = getattr(torch, name)
    with mp.workdps(100):
        return [
            *entropy_rows(args, dtype),
            *multilabel_rows(args, dtype),
            *distillation_rows(args, dtype),
            *barron_rows(args, dtype),
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=DTYPES, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=available_cpus(), help="processes, one dtype each"
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.dtype or DTYPES
    work = functools.partial(dtype_rows, args)
    rows = []
    with contextlib.ExitStack() as stack:
        results: Iterator[list[dict[str, Any]]] = map(work, names)
        if args.jobs > 1:
            context = multiprocessing.get_context("spawn")  # no fork of the parent's threads
            pool = ProcessPoolExecutor(min(args.jobs, len(names)), mp_context=context)
            results = stack.enter_context(pool).map(work, names)
        for name, found in zip(names, results, strict=True):
            rows.extend(found)
            print(
                name,
                len(rows),
                "records",
                sum(row["passed"] is False for row in rows),
                "failures",
                flush=True,
            )
    metadata: dict[str, Any] = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        python=platform.python_version(),
        device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        quick=args.quick,
        total_records=len(rows),
        candidate_checks=sum(r["passed"] is not None for r in rows),
        reference_observations=sum(r["passed"] is None for r in rows),
        failures=sum(r["passed"] is False for r in rows),
    )
    root = Path(__file__).resolve().parents[2]
    metadata["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("tests/python/release_numerics.py", "python/libsdca/torch.py")
    }
    (args.output / "numerics.json").write_text(
        json.dumps(dict(metadata=metadata, rows=rows), indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    raise SystemExit(bool(metadata["failures"]))


if __name__ == "__main__":
    main()

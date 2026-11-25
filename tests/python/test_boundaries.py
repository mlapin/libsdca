"""The boundary scan (boundaries.py) as tests: inside the documented ranges every family passes,
and each documented limit still fails where documented.

- test_scan[<loss>]: the quick scan of one loss (every family, float64 and float32, NumPy and
  the PyTorch CPU kernels where installed, no bisection): a few seconds per loss. Every failure
  must match a documented limit (docs/API.md, Known limits; boundaries.LIMITS_DOCUMENTED) or be
  a PENDING series, and every PENDING series that ran must still fail (a fix removes it).
- test_scan_full[<loss>] (long): the quick scan in every format on every CPU backend.
- test_scan_cuda[<loss>] (cuda): the quick scan on the PyTorch CUDA kernels.
- test_documented_limit[<id>]: one point per documented limit, which must fail with that limit.
"""

from __future__ import annotations

import math
import platform
import sys
from pathlib import Path

import boundaries as B
import pytest

DEFAULT_MODES = ("float64", "float32")
CPU = (B.Backend("numpy", "cpu", "kernel"), B.Backend("torch", "cpu", "kernel"))
FULL = (
    B.Backend("numpy", "cpu", "kernel"),
    *(B.Backend("torch", "cpu", o) for o in ("kernel", "auto", "composed")),
    *(B.Backend("jax", "cpu", o) for o in ("kernel", "auto")),
)
CUDA = (B.Backend("torch", "cuda", "kernel"),)
ARM_MAC = sys.platform == "darwin" and platform.machine() == "arm64"

# Series whose failures match no documented limit (found by the scan; docs/API.md or a fix
# pending): (loss, mode, family, variant). A fix or a documented limit deletes its line.
# (none: the subnormal upstream series are the documented "Gradients near the subnormal
# range")
PENDING: set[tuple[str, str, str, str]] = set()


def scan(label: str, backends: tuple[B.Backend, ...], modes: tuple[str, ...]) -> list[dict]:
    spec = B.SPEC[label]
    chosen = tuple(b for b in backends if not B.unavailable(b))
    if not chosen:
        pytest.skip("no backend available")
    series = []
    for name, family in B.FAMILIES.items():
        if family.applies(spec):
            task = B.Task(
                label, name, B.available_modes(modes), chosen, B.Options(quick=True), refine=0
            )
            series += B.run_task(task)[1]
    return series


def check(series: list[dict]) -> None:
    new = [s for s in series if s.get("new")]
    unexpected = [
        s for s in new if (s["loss"], s["mode"], s["family"], s["variant"]) not in PENDING
    ]
    ran = {(s["loss"], s["mode"], s["family"], s["variant"]) for s in series}
    still = {(s["loss"], s["mode"], s["family"], s["variant"]) for s in new}
    fixed = sorted((ran & PENDING) - still)
    lines = [
        f"{s['loss']} {s['mode']} {s['backend']} {s['family']}:{s['variant']} from "
        f"{s['new']}: {s['repro']}"
        for s in unexpected
    ]
    assert not unexpected, "failures outside the documented limits:\n" + "\n".join(lines)
    assert not fixed, f"PENDING series pass now (delete them from PENDING): {fixed}"


LOSSES = [pytest.param(s.label, id=s.label) for s in B.SPECS]


@pytest.mark.parametrize("label", LOSSES)
def test_scan(label: str) -> None:
    check(scan(label, CPU, DEFAULT_MODES))


@pytest.mark.long
@pytest.mark.parametrize("label", LOSSES)
def test_scan_full(label: str) -> None:
    check(scan(label, FULL, tuple(B.MODES)))


@pytest.mark.cuda
@pytest.mark.parametrize("label", LOSSES)
def test_scan_cuda(label: str) -> None:
    reason = B.unavailable(CUDA[0])
    if reason:
        pytest.skip(reason)
    check(scan(label, CUDA, tuple(B.MODES)))


# One point per documented limit, on NumPy's CPU kernels: (id, loss, mode, family, variant,
# parameter, the limit's key).
F64_MAX = 1.7976931348623157e308
DOCUMENTED = [
    ("topk_ce-k3-near-overflow", "topk_cross_entropy[k=3]", "float64", "spread", "minus-C16",
     F64_MAX, "sums-near-overflow"),
    ("multilabel_ce-extreme", "multilabel_cross_entropy", "float64", "spread", "minus-C3",
     F64_MAX, "extreme-spreads"),
    ("proximal_ce-extreme", "proximal_cross_entropy[alpha=1]", "float64", "spread", "minus-C3",
     F64_MAX, "extreme-spreads"),
    ("proximal_distillation-extreme", "proximal_distillation[T=1,alpha=1]", "float64",
     "distillation", "shift-pair", F64_MAX, "extreme-spreads"),
    ("smoothed_topk_hinge-margins", "topk_hinge_beta[k=2,gamma=0.5]", "float64", "masked",
     "three-bottom", 1e17, "hinge-margins"),
    ("smoothed_multilabel_hinge-near-overflow", "multilabel_hinge[gamma=0.5]", "float64",
     "margin", "top-C10", F64_MAX, "sums-near-overflow"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("label", "mode", "family", "variant", "param", "key"),
    [pytest.param(*case[1:], id=case[0]) for case in DOCUMENTED],
)
def test_documented_limit(
    label: str, mode: str, family: str, variant: str, param: float, key: str
) -> None:
    if key == "sums-near-overflow" and label.startswith("multilabel_hinge") and not ARM_MAC:
        pytest.skip("the smoothed hinges fail near overflow on macOS arm64 only (docs/API.md)")
    spec, fam = B.SPEC[label], B.FAMILIES[family]
    dtype, precision = B.MODES[mode]
    point = fam.build(spec, dtype, B.P.work_dtype(dtype, precision), variant, param)
    assert point is not None
    backend = CPU[0]
    ctx = B.Context(spec, mode, backend, family, variant, point)
    record = B.evaluate(ctx)
    failures = record.get("failures", [])
    assert failures, f"{label} {family}:{variant} at {param} passes: the limit {key} is fixed"
    keys = {k for f in failures for k in f.keys}
    assert key in keys, [f.line() + f" {f.keys}" for f in failures]
    assert all(f.keys for f in failures), [f.line() for f in failures if not f.keys]


def test_grids() -> None:
    """Each scores grid reaches the format's max, in scan order, with the landmarks."""
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        domain = B.Domain(1.0, B.LIMITS[dtype].max)
        marks = B.landmarks(domain, dtype, "float32" if dtype != "float64" else dtype, 10)
        for quick in (True, False):
            xs = B.grid(domain, marks, quick)
            assert xs == sorted(xs) and xs[-1] == B.LIMITS[dtype].max
            assert all(math.isfinite(x) for x in xs)
    down = B.Domain(1.0, B.P.eta("float32"), kind="small")
    xs = B.grid(down, B.landmarks(down, "float32", "float32", 4), False)
    assert xs == sorted(xs, reverse=True) and xs[-1] == B.P.eta("float32")


def test_resume_matches_scan_settings(tmp_path: Path) -> None:
    from dataclasses import replace

    # A task without modes exercises checkpoint writing without invoking a backend.
    task = B.Task("cross_entropy", "margin", (), CPU, B.Options(quick=True), refine=0)
    saved = B.run_task(task, tmp_path)
    assert B._part(tmp_path, task) == saved
    assert B._part(tmp_path, replace(task, refine=8)) is None
    assert B._part(tmp_path, replace(task, build="different")) is None


def test_empty_scan_reports_missing_coverage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no supported points"):
        B.main(
            [
                "--out",
                str(tmp_path),
                "--losses",
                "cross_entropy",
                "--frameworks",
                "numpy",
                "--families",
                "margin",
                "--modes",
                "",
            ]
        )
    assert "No supported points were evaluated" in (tmp_path / "boundaries.md").read_text()
    assert str(tmp_path) not in (tmp_path / "meta.json").read_text()
    report = (tmp_path / "boundaries.md").read_text()
    assert B.build_digest() in report and '"frameworks": ["numpy"]' in report

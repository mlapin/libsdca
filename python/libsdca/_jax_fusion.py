"""Whether the jax entropy and distillation losses fence their passes on the CPU.

XLA:CPU (jaxlib 0.12) un-fuses an (N, C) intermediate that a row reduction and another
operation share into a memory write partitioned over 4 or 5 threads. Fenced, each pass over
the scores above jax._SMALL takes a view of its own behind lax.optimization_barrier: fused
across every thread, it recomputes its exponentials. Which is faster depends on the machine,
not on its platform: on an EPYC 7J13 fences make distillation's forward and backward 1.2x
(8 threads) to 5.6x (64) faster; on an 18-core Apple M5 Pro they make it 1.4-2x slower.

SDCA_JAX_CPU_FUSION selects them, read when a loss is traced (JAX reuses a function's trace
until jax.clear_caches()): "fenced", "unfenced", or "auto" (default, also unset), which
takes the faster variant on this machine per loss family and storage width. auto times both
once (calibrate: WORKLOAD alternately, in a thread of its own on concrete arrays, so a loss
being traced under jit may trigger it) and keeps the fenced one unless the unfenced one
takes at most MARGIN of its time. The decision is cached in SDCA_CACHE_DIR (as _jax_ffi) for
this CPU model, its visible CPUs, XLA_FLAGS, the single-thread settings, jaxlib and libsdca.
Families outside CALIBRATED, and those whose calibration fails, take heuristic.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ENV = "SDCA_JAX_CPU_FUSION"
MODES = ("auto", "fenced", "unfenced")
# distillation_value: distillation_loss without its gradient (jax._distillation_value).
FAMILIES = (
    "cross_entropy",
    "multilabel_cross_entropy",
    "distillation",
    "distillation_value",
    "proximal_distillation",
)
# auto times these. Proximal distillation's solve dominates: fences change its time by at
# most 10% on the machines measured, and timing it would take 2.5-8 s.
CALIBRATED = FAMILIES[:4]
# The calibration call (rows, classes) and its alternating samples, each of as many calls as
# SAMPLE seconds hold (at least one): ROUNDS, or as many as BUDGET seconds hold but at least
# MIN_ROUNDS.
WORKLOAD = (4096, 1024)
ROUNDS, MIN_ROUNDS, SAMPLE, BUDGET = 7, 3, 0.01, 0.3
# auto keeps the fences unless the unfenced median takes at most this fraction of their time.
MARGIN = 0.85
_lock = threading.Lock()
_decisions: dict[str, bool] = {}
_local = threading.local()
_log = logging.getLogger("libsdca")


def mode() -> str:
    """SDCA_JAX_CPU_FUSION as "auto" (also unset), "fenced" or "unfenced"; ValueError else."""
    value = os.environ.get(ENV, "").strip().lower() or "auto"
    if value not in MODES:
        raise ValueError(f"{ENV} must be auto, fenced or unfenced, not {value!r}")
    return value


def fenced(family: str, dtype: Any, single_thread: bool) -> bool:
    """Whether the passes of family (one of FAMILIES) over scores of dtype, traced now for the
    CPU, are fenced."""
    forced = getattr(_local, "forced", None) or mode()
    if forced != "auto":
        return forced == "fenced"
    if family not in CALIBRATED:
        return heuristic(family, single_thread)
    return decision(family, calibration_dtype(dtype), single_thread)


def calibration_dtype(dtype: Any) -> str:
    """The storage width auto times; float64 has its own arithmetic and memory cost."""
    import numpy as np

    width = np.dtype(dtype).itemsize
    return "bfloat16" if width <= 2 else "float32" if width == 4 else "float64"


def heuristic(family: str, single_thread: bool) -> bool:
    """The decision where calibration fails: fenced (they win on every Linux machine
    measured) but the gradient of distillation on one thread, where no pass is partitioned."""
    return not (single_thread and family in ("distillation", "proximal_distillation"))


def decision(family: str, dtype: str, single_thread: bool) -> bool:
    """auto's decision for family at dtype (calibration_dtype): cached in this process, else
    on disk, else calibrated."""
    key = machine_key(single_thread)
    identity = json.dumps([key, family, dtype], sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    with _lock:
        if digest in _decisions:
            return _decisions[digest]
        record = _cache_file(digest)
        try:
            result = json.loads(record.read_text())["fenced"]
            if not isinstance(result, bool):
                raise ValueError("cached fusion decision must be a boolean")
        except (OSError, ValueError, KeyError, TypeError):
            result = _calibrated(family, dtype, single_thread, key, record)
        _decisions[digest] = result
        return result


def _calibrated(
    family: str, dtype: str, single_thread: bool, key: dict[str, Any], record: Path
) -> bool:
    try:
        timing = calibrate(family, dtype)
    except Exception as error:
        result = heuristic(family, single_thread)
        _log.warning(
            "libsdca: %s calibration of %s (%s) failed (%s); %s",
            ENV,
            family,
            dtype,
            error,
            "fenced" if result else "unfenced",
        )
        return result
    result = timing["unfenced_s"] > MARGIN * timing["fenced_s"]
    content = {"key": key, "family": family, "dtype": dtype, "fenced": result, **timing}
    _write(record, content)
    return result


def calibrate(family: str, dtype: str) -> dict[str, Any]:
    """Median seconds (fenced_s, unfenced_s) of a forward and backward (distillation_value: a
    forward) of family at WORKLOAD in dtype, timed alternately in a thread of its own (no
    trace of the caller's is active there), the rounds and the seconds the calibration took."""
    import jax

    outcome: dict[str, Any] = {}
    x64 = jax.config.jax_enable_x64

    def run() -> None:
        try:
            # JAX config contexts are thread-local: carry the caller's precision into
            # the worker instead of silently timing a downcast float64 program.
            with _precision_scope(x64):
                outcome.update(_timed(family, dtype))
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=run, name="libsdca-cpu-fusion", daemon=True)
    worker.start()
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome


def _precision_scope(enabled: bool) -> Any:
    import jax
    import jax.experimental

    scope = getattr(jax, "enable_x64", None)
    if scope is None:
        scope = getattr(jax.experimental, "enable_x64", None)
    if scope is None:
        raise RuntimeError("JAX has no thread-local x64 configuration scope")
    return scope(enabled)


def _timed(family: str, dtype: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    start = time.perf_counter()
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(0)
    rows, classes = WORKLOAD

    def scores() -> Any:
        z = rng.standard_normal((rows, classes), np.float32) * 3
        return jax.device_put(z.astype(jnp.dtype(dtype)), cpu)

    args = [scores()]
    if "distillation" in family:
        args.append(scores())
    elif family == "cross_entropy":
        args.append(jax.device_put(rng.integers(0, classes, rows).astype(np.int32), cpu))
    else:
        mask = rng.random((rows, classes)) < 4 / classes
        mask[np.arange(rows), rng.integers(0, classes, rows)] = True
        args.append(jax.device_put(mask, cpu))
    steps = {variant: _step(family, variant, args) for variant in ("fenced", "unfenced")}
    begin = time.perf_counter()
    for step in steps.values():
        jax.block_until_ready(step(*args))
    pair = time.perf_counter() - begin
    calls = max(1, int(SAMPLE / pair))
    rounds = max(MIN_ROUNDS, min(ROUNDS, int(BUDGET / (calls * pair))))
    times: dict[str, list[float]] = {variant: [] for variant in steps}
    names = list(steps)
    for r in range(rounds):
        for variant in names[:: 1 if r % 2 == 0 else -1]:
            step = steps[variant]
            begin = time.perf_counter()
            for _ in range(calls):
                jax.block_until_ready(step(*args))
            times[variant].append((time.perf_counter() - begin) / calls)
    return {
        **{f"{variant}_s": statistics.median(samples) for variant, samples in times.items()},
        "rounds": rounds,
        "samples_s": times,
        "seconds": time.perf_counter() - start,
    }


def _step(family: str, variant: str, args: list[Any]) -> Callable[..., Any]:
    """A forward and backward of family (the forward of distillation_value), traced with
    variant and compiled for args."""
    import jax

    from . import jax as sdca_jax

    def loss(z: Any, target: Any) -> Any:
        if family.startswith("distillation"):
            return sdca_jax.distillation_loss(z, target, 2.0)
        if family == "proximal_distillation":
            return sdca_jax.proximal_distillation_loss(z, target, 2.0, 1.0)
        return sdca_jax.sdca_loss(z, target, family)

    _local.forced = variant
    try:
        step = loss if family == "distillation_value" else jax.value_and_grad(loss)
        return jax.jit(step).lower(*args).compile()
    finally:
        _local.forced = None


def machine_key(single_thread: bool) -> dict[str, Any]:
    """What the calibration depends on, besides the loss family."""
    import jax

    from . import __version__

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 0
    return {
        "cpu": _cpu_model(),
        "machine": platform.machine(),
        "cpus": cpus,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "single_thread": single_thread,
        "jaxlib": _version("jaxlib"),
        "libsdca": __version__,
        "sources": _source_digest(),
        "x64": bool(jax.config.jax_enable_x64),
        "workload": [*WORKLOAD, ROUNDS, MIN_ROUNDS, SAMPLE, BUDGET, MARGIN],
    }


@functools.cache
def _source_digest() -> str:
    root = Path(__file__).parent
    return hashlib.sha256(
        b"\0".join(
            (root / name).read_bytes() for name in ("jax.py", "_jax_losses.py", "_jax_fusion.py")
        )
    ).hexdigest()


@functools.cache
def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            command = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(
                command, capture_output=True, text=True, check=True
            ).stdout.strip()
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith(("model name", "Model")):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _cache_file(digest: str) -> Path:
    from . import _jax_ffi

    return _jax_ffi.cache_root() / f"jax_cpu_fusion_{digest[:16]}.json"


def _write(record: Path, content: dict[str, Any]) -> None:
    """Write record atomically; a cache that cannot be written only costs a calibration."""
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=record.name + ".", dir=record.parent)
        with os.fdopen(handle, "w") as file:
            json.dump(content, file, indent=1, sort_keys=True)
        os.replace(name, record)
    except OSError as error:
        _log.debug("libsdca: cannot cache %s: %s", record, error)

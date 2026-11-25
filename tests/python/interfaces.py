"""The CPU interfaces of the losses, for the tests that run one input through each of them
(test_cross_interface.py, test_rule_table.py).

- Storage types: float64 values exact in a dtype, their bytes (bfloat16 as uint16 bits).
- The C ABI through ctypes: c_loss_evaluate, with its return code, message and report.
- The compiled interfaces: tests/cpp/cross_interface.cpp (sdca::kernel::evaluate and
  sdca/loss.h) and rust/tests/cross_interface.rs (Loss::loss_and_grad_with), which read a
  directory of cases (cases.txt, raw buffers) and write their outputs next to them.
- shared_directory: one directory per session, prepared once for every xdist worker.

A line of cases.txt (manifest_line): id key kind k gamma temperature label_smoothing ignore
ignore_index dtype n classes target weight threads precision upstream check_rows. The inputs of
a key: <key>.scores and <key>.teacher (the dtype's storage), <key>.labels, <key>.set_labels
and <key>.offsets (int64), <key>.mask (uint8), <key>.probs and <key>.weight (float64). The
outputs of a case: <id>.<interface>.values (float64), .grads (storage), and
<id>.cpp_kernel.status ("status invalid_rows first_invalid_row"), <id>.cpp_loss.error,
<id>.rust.report ("invalid_rows first_invalid_row", -1 for none) or <id>.rust.error.
"""

from __future__ import annotations

import contextlib
import ctypes as ct
import dataclasses
import functools
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = ROOT / "tests/cpp/cross_interface.cpp"
RUST_MANIFEST = ROOT / "rust/Cargo.toml"
NATIVE_DTYPES = ("float32", "float64")
# The bytes the C++ helper writes into its outputs before the call (check_rows: unchanged).
UNWRITTEN = 0xAB

# --- storage types ---------------------------------------------------------------------------


def bfloat16_bits(x: np.ndarray) -> np.ndarray:
    """float32 values rounded to bfloat16 (to nearest, ties to even), as their bits."""
    f = np.asarray(x, np.float32)
    bits = f.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(f), ((bits >> 16) | 0x40).astype(np.uint16), rounded)


def from_bfloat16_bits(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def represent(x: np.ndarray, dtype: str) -> np.ndarray:
    """float64 values of x rounded to dtype."""
    if dtype == "bfloat16":
        return from_bfloat16_bits(bfloat16_bits(x.astype(np.float32)))
    return x.astype(dtype).astype(np.float64)


def storage(x: np.ndarray, dtype: str) -> np.ndarray:
    """x (exact in dtype) in dtype's storage (bfloat16 as uint16 bits)."""
    x = np.asarray(x)
    return bfloat16_bits(x) if dtype == "bfloat16" else x.astype(dtype)


def storage_dtype(dtype: str) -> np.dtype:
    return np.dtype(np.uint16) if dtype == "bfloat16" else np.dtype(dtype)


def values_of(data: bytes, dtype: str) -> np.ndarray:
    """float64 values of a buffer of dtype's storage."""
    a = np.frombuffer(data, storage_dtype(dtype))
    return from_bfloat16_bits(a) if dtype == "bfloat16" else a.astype(np.float64)


# --- the C ABI -------------------------------------------------------------------------------


class _Loss(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("kind", ct.c_int),
        ("k", ct.c_size_t),
        ("gamma", ct.c_double),
        ("temperature", ct.c_double),
        ("weight", ct.c_void_p),
        ("label_smoothing", ct.c_double),
        ("ignore", ct.c_int),
        ("ignore_index", ct.c_int64),
    ]


class _Report(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("invalid_rows", ct.c_size_t),
        ("first_invalid_row", ct.c_size_t),
    ]


class _Execution(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("precision", ct.c_int),
        ("num_threads", ct.c_size_t),
        ("check_rows", ct.c_int),
        ("report", ct.POINTER(_Report)),
    ]


class _Targets(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("kind", ct.c_int),
        ("labels", ct.c_void_p),
        ("offsets", ct.c_void_p),
        ("mask", ct.c_void_p),
        ("probabilities", ct.c_void_p),
        ("logits", ct.c_void_p),
    ]


@functools.cache
def c_library() -> ct.CDLL:
    from libsdca._native import _library_path, _load_blas_package

    path = _library_path()
    _load_blas_package(path)
    lib = ct.CDLL(str(path))
    lib.sdca_loss_evaluate.restype = ct.c_int
    lib.sdca_loss_evaluate.argtypes = [
        ct.POINTER(_Loss),
        ct.POINTER(_Execution),
        ct.c_int,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.POINTER(_Targets),
        ct.c_void_p,
        ct.c_void_p,
    ]
    lib.sdca_last_error.restype = ct.c_char_p
    return lib


SDCA_OK, SDCA_INVALID_ROWS, SDCA_INVALID_ARGUMENT = 0, 1, -2


@dataclasses.dataclass(frozen=True)
class CResult:
    code: int
    message: str  # sdca_last_error()
    invalid_rows: int
    first_invalid_row: int
    values: np.ndarray  # float64 (n,)
    grads: np.ndarray | None  # the storage dtype (n, classes)


def label_sets(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The labels and offsets of the label sets of a mask."""
    flat = [np.flatnonzero(row) for row in mask]
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in flat])]).astype(np.int64)
    return np.concatenate([*flat, np.zeros(0, np.int64)]).astype(np.int64), offsets


def c_loss_evaluate(
    kind: int,
    dtype: str,
    scores: np.ndarray,
    target_kind: str,
    target: np.ndarray,
    *,
    k: int = 0,
    gamma: float = 0.0,
    temperature: float = 1.0,
    threads: int = 1,
    precision: int = 0,
    gradients: bool = True,
    check_rows: bool = False,
    fill: float = math.nan,
    weight: np.ndarray | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int | None = None,
) -> CResult:
    """sdca_loss_evaluate through ctypes, with the options of struct sdca_loss (weight,
    label_smoothing, ignore_index). target_kind: labels, sets or mask (of a mask target), probs
    or teacher. The outputs start as fill: an error leaves them so."""
    n, classes = scores.shape
    ignore = ignore_index is not None
    w = None if weight is None else np.ascontiguousarray(weight, np.float64)
    loss = _Loss(
        ct.sizeof(_Loss),
        kind,
        k,
        gamma,
        temperature,
        None if w is None else w.ctypes.data,
        label_smoothing,
        int(ignore),
        ignore_index if ignore else -100,
    )
    report = _Report(ct.sizeof(_Report), 0, 0)
    execution = _Execution(
        ct.sizeof(_Execution), precision, threads, int(check_rows), ct.pointer(report)
    )
    z = np.ascontiguousarray(scores, dtype)
    keep: list[np.ndarray] = []
    t = _Targets(ct.sizeof(_Targets))

    def address(a: np.ndarray) -> int:
        keep.append(np.ascontiguousarray(a))
        return keep[-1].ctypes.data

    if target_kind == "labels":
        t.kind, t.labels = 0, address(target.astype(np.int64))
    elif target_kind == "sets":
        labels, offsets = label_sets(target)
        t.kind, t.labels, t.offsets = 1, address(labels), address(offsets)
    elif target_kind == "mask":
        t.kind, t.mask = 2, address(target.astype(np.uint8))
    elif target_kind == "probs":
        t.kind, t.probabilities = 3, address(target.astype(np.float64))
    else:
        t.kind, t.logits = 4, address(target.astype(dtype))
    values = np.full(n, fill)
    grads = np.full_like(z, fill) if gradients else None
    code = c_library().sdca_loss_evaluate(
        ct.byref(loss),
        ct.byref(execution),
        NATIVE_DTYPES.index(dtype),
        n,
        classes,
        z.ctypes.data,
        ct.byref(t),
        values.ctypes.data,
        None if grads is None else grads.ctypes.data,
    )
    message = (c_library().sdca_last_error() or b"").decode()
    return CResult(code, message, report.invalid_rows, report.first_invalid_row, values, grads)


# --- the compiled interfaces -----------------------------------------------------------------


class Missing(Exception):
    """A tool that a compiled interface needs is not installed: its cases skip."""


def _cpp_binary() -> Path:
    """The C++ helper, compiled once per source, headers, flags and compiler."""
    from libsdca import _extensions

    reason = _extensions.missing_compiler()
    if reason:
        raise Missing(reason)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fno-fast-math", "-ffp-contract=off", "-pthread"]
    compiler = _extensions.compiler_command()
    headers = sorted((ROOT / "include/sdca").rglob("*.h"))
    parts = [CPP_SOURCE.read_bytes(), *(h.read_bytes() for h in headers), *map(str.encode, flags)]
    parts.append(_extensions._compiler_version(tuple(compiler)).encode())
    digest = hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]
    binary = _extensions.cache_root() / f"cross_interface-{digest}" / "cross_interface"
    if binary.is_file():
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    private = binary.with_name(f"cross_interface.{os.getpid()}")
    command = [*compiler, *flags, f"-I{ROOT / 'include'}", str(CPP_SOURCE), "-o", str(private)]
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode:
        raise RuntimeError(f"tests/cpp/cross_interface.cpp did not build:\n{built.stderr[-4000:]}")
    private.replace(binary)
    return binary


def run_cpp(directory: Path) -> str:
    """Run the C++ helper on a directory of cases: "" or how it failed."""
    try:
        binary = _cpp_binary()
    except RuntimeError as error:
        return str(error)
    ran = subprocess.run([binary, directory], capture_output=True, text=True, check=False)
    return f"cross_interface failed:\n{ran.stderr[-4000:]}" if ran.returncode else ""


def run_rust(directory: Path) -> str:
    """Run the Rust helper on a directory of cases: "" or how it failed."""
    cargo = shutil.which("cargo")
    if cargo is None:
        raise Missing("cargo is not on PATH")
    command = [cargo, "test", "-q", "--manifest-path", str(RUST_MANIFEST)]
    command += ["--test", "cross_interface"]
    env = {**os.environ, "SDCA_CROSS_INTERFACE_DIR": str(directory)}
    ran = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if ran.returncode:
        return f"cargo test --test cross_interface failed:\n{(ran.stdout + ran.stderr)[-4000:]}"
    return ""


def manifest_line(
    id: str,
    key: str,
    kind: int,
    dtype: str,
    n: int,
    classes: int,
    target: str,
    *,
    k: int = 1,
    gamma: float = 0.0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    ignore_index: int | None = None,
    weight: bool = False,
    threads: int = 0,
    precision: int = 0,
    upstream: float = 1.0,
    check_rows: bool = False,
) -> str:
    """A line of cases.txt."""
    ignore = ignore_index is not None
    fields = [id, key, kind, k, repr(gamma), repr(temperature), repr(label_smoothing)]
    fields += [int(ignore), ignore_index if ignore else -100, dtype, n, classes, target]
    fields += [int(weight), threads, precision, repr(upstream), int(check_rows)]
    return " ".join(str(v) for v in fields)


def write_inputs(
    directory: Path,
    key: str,
    dtype: str,
    scores: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    teacher: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> None:
    """The input files of a key (the arrays given; labels default to zeros)."""

    def put(suffix: str, array: np.ndarray) -> None:
        (directory / f"{key}.{suffix}").write_bytes(np.ascontiguousarray(array).tobytes())

    put("scores", storage(scores, dtype))
    put("labels", np.zeros(len(scores), np.int64) if labels is None else labels.astype(np.int64))
    if weight is not None:
        put("weight", weight.astype(np.float64))
    if mask is not None:
        put("mask", mask.astype(np.uint8))
        set_labels, offsets = label_sets(mask)
        put("set_labels", set_labels)
        put("offsets", offsets)
    if probs is not None:
        put("probs", probs.astype(np.float64))
    if teacher is not None:
        put("teacher", storage(teacher, dtype))


def run_compiled(directory: Path) -> dict[str, dict[str, str]]:
    """Run the C++ and Rust helpers: {"missing": {name: tool}, "failed": {name: how}}."""
    outcome: dict[str, dict[str, str]] = {"missing": {}, "failed": {}}
    for name, run in (("cpp", run_cpp), ("rust", run_rust)):
        try:
            reason = run(directory)
        except Missing as missing:
            outcome["missing"][name] = str(missing)
            continue
        if reason:
            outcome["failed"][name] = reason
    return outcome


@dataclasses.dataclass(frozen=True)
class Session:
    directory: Path
    missing: dict[str, str]  # cpp or rust -> the tool it lacks (its cases skip)
    failed: dict[str, str]  # cpp or rust -> how it failed to build or run (its cases fail)

    def require(self, interface: str) -> None:
        """Skip where a compiled interface lacks a tool; fail where it did not run."""
        group = "cpp" if interface.startswith("cpp") else interface
        if group in self.missing:
            pytest.skip(f"{interface} cannot run here: {self.missing[group]}")
        if group in self.failed:
            pytest.fail(f"{interface} did not run: {self.failed[group]}")

    def path(self, id: str, interface: str, part: str) -> Path:
        return self.directory / f"{id}.{interface}.{part}"


def shared_session(
    tmp_path_factory: pytest.TempPathFactory, name: str, prepare: Callable[[Path], None]
) -> Session:
    """A directory written by prepare and read by the compiled interfaces, once per session
    for every xdist worker."""
    root = tmp_path_factory.getbasetemp()
    root = root.parent if os.environ.get("PYTEST_XDIST_WORKER") else root
    directory, done = root / name, root / f"{name}.json"
    with open(root / f"{name}.lock", "w") as lock:
        with contextlib.suppress(ImportError):
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        if not done.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            prepare(directory)
            done.write_text(json.dumps(run_compiled(directory)))
    outcome = json.loads(done.read_text())
    return Session(directory, outcome["missing"], outcome["failed"])


# --- the frameworks --------------------------------------------------------------------------


@contextlib.contextmanager
def torch_threads(torch: Any, threads: int) -> Iterator[None]:
    """torch's intra-op threads (and so its CPU kernels') set to threads (0: unchanged)."""
    before = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(before)


@contextlib.contextmanager
def x64(jax: Any, enabled: bool) -> Iterator[None]:
    context = getattr(jax, "enable_x64", None)
    if context is None:
        context = importlib.import_module("jax.experimental").enable_x64
    with context(enabled):
        yield


# --- the losses by their shared names (masked_cases: topk_hinge_alpha and topk_hinge_beta,
# cross_entropy_probabilities) --------------------------------------------------------------

PROBABILITIES = "cross_entropy_probabilities"
DISTILLATION = ("distillation", "proximal_distillation")
C_KINDS = {
    "cross_entropy": 0,
    PROBABILITIES: 0,
    "topk_cross_entropy": 1,
    "multiclass_hinge": 2,
    "topk_hinge_alpha": 3,
    "topk_hinge_beta": 4,
    "multilabel_hinge": 5,
    "multilabel_cross_entropy": 6,
    "proximal_cross_entropy": 7,
    "distillation": 8,
    "proximal_distillation": 9,
}


def c_parameters(loss: str, params: dict[str, float]) -> dict[str, Any]:
    """The k, gamma and temperature of a loss's sdca_loss: only those it takes (SDCA_LOSS_INIT
    for the others); alpha is the step (gamma) of the proximal losses."""
    kind = C_KINDS[loss]
    out: dict[str, Any] = {"k": int(params.get("k", 0)) if kind in (1, 3, 4) else 0}
    out["gamma"] = params.get("alpha", 1.0) if kind in (7, 9) else params.get("gamma", 0.0)
    out["temperature"] = params.get("temperature", 1.0) if kind in (8, 9) else 1.0
    return out


def numpy_loss(loss: str, params: dict[str, float]) -> Any:
    """The NumPy loss object of a loss and its parameters."""
    import named_losses

    name = "cross_entropy" if loss == PROBABILITIES else loss
    return named_losses.numpy_loss(name, **params)


def framework_call(
    module: Any, loss: str, logits: Any, target: Any, params: dict[str, float], **options: Any
) -> Any:
    """The loss of libsdca.torch or libsdca.jax (module) with its parameters and options
    (reduction, backend, weight, ignore_index, label_smoothing, ...), classes last."""
    import named_losses

    if loss in DISTILLATION:
        temperature = params.get("temperature", 1.0)
        if loss == "distillation":
            return module.distillation_loss(logits, target, temperature=temperature, **options)
        step = params.get("alpha", 1.0)
        return module.proximal_distillation_loss(
            logits, target, temperature=temperature, gamma=step, **options
        )
    name = "cross_entropy" if loss == PROBABILITIES else loss
    k, gamma, alpha = int(params.get("k", 1)), params.get("gamma", 0.0), params.get("alpha", 1.0)
    return named_losses.loss(module, name, logits, target, k=k, gamma=gamma, alpha=alpha, **options)

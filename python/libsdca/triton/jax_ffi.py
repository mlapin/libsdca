"""Compile a shared Triton plan into one CUDA XLA FFI call. Array arguments name XLA
buffers and scalar arguments travel in the launch program (see _jax_cuda.cpp)."""

from __future__ import annotations

import base64
import ctypes
import dataclasses
import functools
import hashlib
from pathlib import Path
from typing import Any

import jax
import numpy as np
import triton
from triton.backends.compiler import GPUTarget

from .. import _extensions
from .._jax_ops import _build, cuda_architecture
from .common import Launch, Plan, Ref, direct_launch

_TARGET = "libsdca_cuda_kernels"
_TYPES = {
    "bool": "i1",
    "uint8": "u8",
    "int32": "i32",
    "uint32": "u32",
    "int64": "i64",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float64": "fp64",
}


def _register(path: Path) -> ctypes.CDLL:
    from jaxlib import xla_client

    library = ctypes.CDLL(str(path))
    stages = {
        stage: jax.ffi.pycapsule(getattr(library, f"sdca_jax_cuda_{stage}"))
        for stage in ("initialize", "execute")
    }
    try:
        jax.ffi.register_ffi_target(
            _TARGET,
            stages,
            platform="CUDA",
            traits=xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE,
        )
    except RuntimeError as error:
        if "does not support custom call traits" not in str(error):
            raise
        # Older PJRT GPU plugins run an execute-only FFI outside command buffers.
        jax.ffi.register_ffi_target(_TARGET, stages["execute"], platform="CUDA")
    return library


@functools.cache
def available() -> str | None:
    build = dataclasses.replace(
        _build(),
        source="_jax_cuda.cpp",
        env="SDCA_JAX_CUDA_FFI",
        what="the CUDA launcher of libsdca.jax",
        link=lambda: ["-ldl"],
    )
    library, reason = _extensions.load(build, _register)
    return None if library is not None else reason


def _compile(launch: Launch, buffers: dict[str, tuple[int, str]], architecture: int):
    """A cubin and its driver arguments in the signature order. XLA aligns buffers to 16 bytes."""
    bound = dict(launch.args)
    options = dict(launch.meta)
    signature, constexprs, attributes = {}, {}, {}
    arguments = []
    for index, param in enumerate(launch.kernel.params):
        name = param.name
        if param.is_constexpr:
            signature[name] = "constexpr"
            constexprs[name] = options.pop(name) if name in options else param.default
        elif isinstance(value := bound[name], Ref):
            buffer, dtype = buffers[value.name]
            signature[name] = "*" + _TYPES[dtype]
            attributes[(index,)] = [["tt.divisibility", 16]]
            arguments.append((0, buffer))
        else:
            wide = not -(2**31) <= value < 2**31
            signature[name] = "i64" if wide else "i32"
            arguments.append((2 if wide else 1, value))
    src = triton.compiler.ASTSource(launch.kernel, signature, constexprs, attributes)
    compiled = triton.compile(src, target=GPUTarget("cuda", architecture, 32), options=options)
    m = compiled.metadata
    if not direct_launch(m):
        raise NotImplementedError(
            f'libsdca: unsupported launch metadata for {m.name}; use backend="composed"'
        )
    # Triton adds up to two scratch pointers after the explicit parameters (unused here).
    arguments.extend(((3, 0), (3, 0)))
    return compiled.asm["cubin"], m, arguments


@functools.lru_cache(maxsize=256)
def _program(plan: Plan, layout: tuple[tuple[str, str], ...], architecture: int):
    buffers = {name: (i, dtype) for i, (name, dtype) in enumerate(layout)}
    table, program = [], []
    binaries = bytearray()
    for i, launch in enumerate(plan.launches):
        image, meta, arguments = _compile(launch, buffers, architecture)
        name = meta.name.encode()
        key = int.from_bytes(hashlib.sha256(image + name).digest()[:8], "little", signed=True)
        blob = base64.b64encode(image)
        offset = len(binaries)
        table.extend(
            (key, offset, len(blob), offset + len(blob), len(name), meta.num_warps, meta.shared)
        )
        binaries.extend(blob + name)
        program.extend((i, *(*launch.grid, 1, 1)[:3], len(arguments)))
        for arg in arguments:
            program.extend(arg)
    return dict(
        kernels=np.asarray(table, np.int64),
        program=np.asarray(program, np.int64),
        binaries=binaries.decode("ascii"),
    )


def run(plan: Plan, env: dict[str, Any], inplace: bool, struct) -> dict[str, Any]:
    outputs = {a.name: struct(a) for a in plan.arrays}
    inputs = {
        v.name: env[v.name]
        for launch in plan.launches
        for _, v in launch.args
        if isinstance(v, Ref) and v.name not in outputs
    }
    aliases = {}
    if inplace:
        aliases[list(inputs).index("Z")] = len(outputs)
        outputs["G"] = jax.ShapeDtypeStruct(env["Z"].shape, env["Z"].dtype)
    layout = tuple((name, a.dtype.name) for name, a in (*inputs.items(), *outputs.items()))
    arch = cuda_architecture()
    if not arch:
        raise NotImplementedError(
            'libsdca: CUDA FFI needs one local architecture; use backend="composed"'
        )
    attrs = _program(plan, layout, arch)
    call = jax.ffi.ffi_call(_TARGET, list(outputs.values()), input_output_aliases=aliases)
    # Internal row values and counted divisors can be float64 even when the user's
    # arrays are restricted to float32; neither escapes this reduction.
    with jax.enable_x64(True):
        results = call(*inputs.values(), **attrs)
    env.update(zip(outputs, results, strict=True))
    return env

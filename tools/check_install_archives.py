"""Check shared CPU sources, storage headers and guides in release archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

REQUIRED = (
    "_cpu_distill.h",
    "_cpu_lanes.h",
    "_cpu_prox.h",
    "_cpu_prox_distill.h",
    "_jax_cpu.py",
    "_jax_ffi.cpp",
    "_torch_entropy.cpp",
)


def files(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
    with tarfile.open(path) as archive:
        result = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                result[member.name.split("/", 1)[1]] = stream.read()
        return result


def check(path: Path, source: Path) -> dict[str, object]:
    content = files(path)
    wheel = path.suffix == ".whl"
    prefix = "libsdca/" if wheel else "python/libsdca/"
    for name in REQUIRED:
        assert prefix + name in content, f"{path.name}: missing {prefix + name}"
    for name, data in content.items():
        if name.startswith(prefix) and name.endswith((".h", ".cpp")):
            for dependency in re.findall(rb'#include\s+"(_(?:cpu|torch)_[^"/]+\.h)"', data):
                included = prefix + dependency.decode() in content
                source_fallback = (
                    not wheel
                    and dependency == b"_cpu_storage.h"
                    and "include/sdca/math/storage.h" in content
                )
                assert included or source_fallback, (name, dependency.decode())
    canonical = source / "include/sdca/math/storage.h"
    storage = "not present in this source revision"
    if canonical.exists():
        name = prefix + "_cpu_storage.h" if wheel else "include/sdca/math/storage.h"
        assert name in content, f"{path.name}: missing storage header {name}"
        assert content[name] == canonical.read_bytes(), f"{name} differs from its canonical source"
        storage = hashlib.sha256(content[name]).hexdigest()
    if not wheel:
        support = source / "tests/python/conformance"
        required = [p for p in support.rglob("*") if p.suffix in {".py", ".json"}]
        required.append(source / "tests/python/release_numerics.py")
        required.extend((source / "docs").rglob("*.md"))
        for support_path in required:
            name = support_path.relative_to(source).as_posix()
            assert name in content, f"{path.name}: missing installed-test support {name}"
            assert content[name] == support_path.read_bytes(), f"{name} differs from its source"
    if wheel:
        record_name = next(name for name in content if name.endswith(".dist-info/RECORD"))
        recorded = {row[0] for row in csv.reader(io.StringIO(content[record_name].decode()))}
        for name in content:
            if name.startswith(prefix) and (
                name.endswith(".h") or name in {prefix + item for item in REQUIRED}
            ):
                assert name in recorded, f"{name} is missing from wheel RECORD"
        assert any("/_libsdca_native" in name and name.endswith(".so") for name in content), path
    if wheel and "linux" in path.name:
        assert any("/_jax_cuda_native" in name and name.endswith(".so") for name in content), path
    return {
        "archive": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "storage_header_sha256": storage,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([check(path, args.source) for path in args.archives], indent=2))


if __name__ == "__main__":
    main()

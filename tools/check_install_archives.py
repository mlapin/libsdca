"""Check shared CPU sources, storage headers and guides in release archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

REQUIRED = (
    "_frontend_cpu.h",
    "_jax_cpu.cpp",
    "_torch_cpu.cpp",
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
            assert (
                member.uid == member.gid == 0
                and member.uname in ("", "root")
                and member.gname in ("", "root", "wheel")
            ), f"{path.name}: build account metadata in tar header for {member.name}"
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                result[member.name.split("/", 1)[1]] = stream.read()
        return result


def check(path: Path, source: Path) -> dict[str, object]:
    content = files(path)
    wheel = path.suffix == ".whl"
    prefix = "libsdca/" if wheel else "python/libsdca/"
    package = source / "python/libsdca"
    for module in sorted(package.rglob("*")):
        if (
            module.suffix not in {".py", ".h", ".cpp"}
            or "include" in module.relative_to(package).parts
        ):
            continue
        name = prefix + module.relative_to(package).as_posix()
        assert name in content, f"{path.name}: missing {name}"
        assert content[name] == module.read_bytes(), f"{path.name}: {name} differs from its source"
    for name in REQUIRED:
        assert prefix + name in content, f"{path.name}: missing {prefix + name}"
    # The shared CPU kernels' headers: libsdca/include/sdca/... in a wheel, include/sdca/... in
    # a source archive, each equal to its source.
    include = prefix + "include/" if wheel else "include/"
    for header in sorted((source / "include/sdca").rglob("*.h")):
        name = include + header.relative_to(source / "include").as_posix()
        assert name in content, f"{path.name}: missing {name}"
        assert content[name] == header.read_bytes(), f"{name} differs from its source"
    for name, data in content.items():
        if name.startswith(prefix) and name.endswith((".h", ".cpp")):
            for dependency in re.findall(rb'#include\s+"((?:_[^"/]+|sdca/[^"]+)\.h)"', data):
                found = dependency.decode()
                target = prefix + found if found.startswith("_") else include + found
                assert target in content, (name, found)
    storage = hashlib.sha256((source / "include/sdca/core/storage.h").read_bytes()).hexdigest()
    if not wheel:
        if (source / ".git").exists():
            names = subprocess.check_output(
                ["git", "-C", str(source), "ls-files", "-z", "--", "tests/python"]
            )
            tests = (source / name.decode() for name in names.split(b"\0") if name)
        else:
            tests = (source / "tests/python").rglob("*")
        required = [path for path in tests if path.suffix in {".py", ".json"}]
        required.extend(
            source / name
            for name in (
                "src/c_api.cpp",
                "src/blas_probe.cpp",
                "src/native_build.txt",
                "sdca_build.py",
                "setup.py",
                "pyproject.toml",
                "VERSION",
            )
        )
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

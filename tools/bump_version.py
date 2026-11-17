"""Set the release version: python tools/bump_version.py 0.3.2 [--date YYYY-MM-DD]

VERSION is the single source that CMake, setup.py, python/build.py, pyproject.toml and
matlab/make.m read. This script also rewrites the copies that cannot read it: the crate
manifest, the Python package, the citation and the release links of the package pages.
A CHANGELOG.md heading "## Unreleased" becomes "## <version> (<date>)"; otherwise add that
heading by hand. tools/quality.py checks that every copy agrees.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINKS = ("pyproject.toml", "docs/PYTHON.md", "docs/RUST.md")


def substitute(path: str, pattern: str, replacement: str, root: Path) -> None:
    file = root / path
    text, count = re.subn(pattern, replacement, file.read_text(), count=1, flags=re.M)
    if not count:
        raise SystemExit(f"{path}: no match for {pattern}")
    file.write_text(text)


def bump(version: str, released: str, root: Path = ROOT) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-.]?\w+)?", version):
        raise SystemExit(f"not a version: {version}")
    date.fromisoformat(released)
    old = (root / "VERSION").read_text().strip()
    (root / "VERSION").write_text(version + "\n")
    substitute("rust/Cargo.toml", r'^version = "[^"]+"', f'version = "{version}"', root)
    substitute(
        "python/libsdca/__init__.py", r'^__version__ = "[^"]+"', f'__version__ = "{version}"', root
    )
    substitute("CITATION.cff", r"^version: \S+$", f"version: {version}", root)
    substitute("CITATION.cff", r"^date-released: \S+$", f"date-released: {released}", root)
    for path in LINKS:
        file = root / path
        text = re.sub(
            rf"(mlapin/libsdca(?:/blob/| --tag ))v{re.escape(old)}\b", rf"\g<1>v{version}",
            file.read_text(),
        )  # fmt: skip
        file.write_text(text)
    changelog = root / "CHANGELOG.md"
    text = changelog.read_text()
    changelog.write_text(text.replace("\n## Unreleased\n", f"\n## {version} ({released})\n", 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("version")
    parser.add_argument("--date", default=date.today().isoformat(), help="release date")
    args = parser.parse_args()
    bump(args.version, args.date)


if __name__ == "__main__":
    main()

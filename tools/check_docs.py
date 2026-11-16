"""Check maintained Markdown for broken local links, unsupported GitHub math and links
into the technical report (the report is linked as a whole: no anchor, no table number).

python tools/check_docs.py
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def markdown_paths(root: Path) -> list[Path]:
    """Public guides and examples; exclude build products and vendored archives."""
    paths = list(root.glob("*.md"))
    for directory in ("docs", "examples"):
        paths.extend((root / directory).rglob("*.md"))
    for directory in ("python", "rust", "matlab", "tools"):
        paths.extend((root / directory).glob("*.md"))
    return sorted(paths)


def prose(text: str) -> str:
    """Mask fenced code while retaining line numbers for link diagnostics."""
    return re.sub(
        r"^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$",
        lambda match: "\n" * match[0].count("\n"),
        text,
        flags=re.M | re.S,
    )


def anchors(text: str) -> set[str]:
    """GitHub heading slugs (including duplicate suffixes) and explicit HTML anchors."""
    found = set(re.findall(r'<(?:a|h[1-6])\b[^>]*\b(?:id|name)="([^"]+)"', text))
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(text), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        candidate, suffix = slug, 0
        while candidate in found:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        found.add(candidate)
    return found


def check_docs(root: Path = ROOT) -> None:
    failures: list[str] = []
    for path in markdown_paths(root):
        text = path.read_text()
        name = path.relative_to(root)
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r"\\operatorname\b", line):
                failures.append(f"{name}:{number}: unsupported math command; use \\mathrm")
            if re.search(r"/libsdca/report/?#", line):
                failures.append(f"{name}:{number}: link to the report without an anchor")
        body = prose(text)
        for match in re.finditer(r"\[[^\]\n]*\]\(([^\s)]+)\)", body):
            link = urlsplit(match[1].strip("<>"))
            if link.scheme or link.netloc:
                continue
            target = path.parent / unquote(link.path) if link.path else path
            number = body.count("\n", 0, match.start()) + 1
            if not target.exists():
                failures.append(f"{name}:{number}: missing local target {match[1]}")
            elif link.fragment and target.suffix == ".md":
                if unquote(link.fragment) not in anchors(target.read_text()):
                    failures.append(f"{name}:{number}: missing heading {match[1]}")
    if failures:
        raise SystemExit("Documentation check failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    check_docs()
    print("Documentation links and math: OK")

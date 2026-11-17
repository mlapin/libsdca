"""Check the links of every tracked Markdown file, unsupported GitHub math and links into
the technical report (the report is linked as a whole: no anchor, no table number).

python tools/check_docs.py              # offline: files, headings and repository URLs
python tools/check_docs.py --external   # also request every external URL

Offline, a link must name a tracked file with the exact case (macOS forgives case, GitHub
does not) and, for Markdown, an existing heading. Links to this repository on GitHub are
resolved in the checkout when they name the default branch or the current version tag.
Guides that also render on PyPI and crates.io must use absolute links.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = re.compile(
    r"https://github\.com/mlapin/libsdca/(?:blob|tree|raw)/([^/#?]+)/?([^#?]*)(?:\?[^#]*)?(?:#(.*))?$"
)
REPOSITORY_BRANCH = "master"
# Package pages render these guides away from the repository, so relative links break.
PACKAGE_PAGES = ("docs/PYTHON.md", "docs/RUST.md")
# Archived run records keep the links of the tree they were made from.
ARCHIVE = "/".join(("private", "experiments")) + "/"
# Metadata that carries URLs but not Markdown links.
METADATA = ("pyproject.toml", "CITATION.cff", "rust/Cargo.toml")
# Statuses of hosts that refuse scripted requests: a warning, not a failure.
UNVERIFIABLE = {401, 403, 429}


def tracked_files(root: Path) -> set[str] | None:
    """Tracked and new (not ignored) paths relative to `root`, or None outside a Git
    checkout (an sdist)."""
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    try:
        listing = subprocess.run(command, cwd=root, capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = {path for path in listing.stdout.split("\0") if path and (root / path).exists()}
    return paths or None


def markdown_paths(root: Path, tracked: set[str] | None) -> list[Path]:
    """Every tracked Markdown file outside vendored code and archived experiment records
    (snapshots of older trees and report sources with generated anchors)."""
    if tracked is None:
        found = list(root.glob("*.md"))
        for directory in ("docs", "examples", "python", "rust", "matlab", "tools"):
            found.extend((root / directory).rglob("*.md"))
        return sorted(path for path in found if "build" not in path.relative_to(root).parts)
    return sorted(
        root / path
        for path in tracked
        if path.endswith(".md") and not path.startswith(("vendor/", ARCHIVE))
    )


def _blank(match: re.Match) -> str:
    return re.sub(r"[^\n]", " ", match[0])


def mask_blocks(text: str) -> str:
    """Blank fenced code and HTML comments, keeping offsets and line numbers."""
    fence = r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\1[`~]*[ \t]*$|\Z)"
    text = re.sub(fence, _blank, text, flags=re.M | re.S)
    return re.sub(r"<!--.*?-->", _blank, text, flags=re.S)


def mask(text: str) -> str:
    """Blank fenced code, HTML comments and inline code."""
    return re.sub(r"(`+)(?!`).*?(?<!`)\1(?!`)", _blank, mask_blocks(text), flags=re.S)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: rendered text, lower case, punctuation dropped."""
    heading = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    heading = re.sub(r"<[^>]+>", "", heading)  # inline HTML
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """GitHub heading slugs (including duplicate suffixes), explicit HTML anchors and the
    Pandoc `{#id}` attributes of documents rendered to HTML."""
    found = set(re.findall(r'<[a-z][a-z0-9]*\b[^>]*\b(?:id|name)="([^"]+)"', text))
    found.update(re.findall(r"\{[^}\n]*#([\w:.-]+)[^}\n]*\}", mask(text)))
    counts: dict[str, int] = {}
    headings = r"^ {0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$"
    for heading in re.findall(headings, mask_blocks(text), re.M):
        base = slug(re.sub(r"\s*\{[^}]*\}$", "", heading))
        count = counts.get(base, 0)
        counts[base] = count + 1
        found.add(f"{base}-{count}" if count else base)
    return found


# Inline links and images, allowing one level of balanced parentheses in the destination.
INLINE = re.compile(
    r"\]\(\s*(<[^>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
DEFINITION = re.compile(r"^ {0,3}\[([^\]^][^\]]*)\]:[ \t]*(<[^>\n]*>|\S+)", re.M)
REFERENCE = re.compile(r"(?<![\]\\])\[((?:[^\[\]\n]|\[[^\]\n]*\])+)\]\[([^\]\n]*)\](?![(:])")
HTML = re.compile(r"""<(?:a|img|source)\b[^>]*?\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)
BARE = re.compile(r"""https?://[^\s<>()\[\]"'`|\\]+""")


def links(text: str) -> list[tuple[int, str]]:
    """(offset, destination) of every link, image, reference definition, HTML link and URL."""
    body = mask(text)
    found = [(match.start(1), match[1].strip("<>")) for match in INLINE.finditer(body)]
    found += [(match.start(2), match[2].strip("<>")) for match in DEFINITION.finditer(body)]
    found += [(match.start(1), match[1]) for match in HTML.finditer(body)]
    taken = {offset for offset, _ in found} | {offset + 1 for offset, _ in found}
    for match in BARE.finditer(body):  # autolinks and bare URLs outside link syntax
        if match.start() not in taken:
            found.append((match.start(), match[0].rstrip(".,;:!?*_~")))
    return sorted(found)


def undefined_references(text: str) -> list[tuple[int, str]]:
    """`[text][label]` whose label has no definition (GitHub renders the brackets)."""
    math = r"\$\$.*?\$\$|\$[^$\n]+\$"
    body = re.sub(math, _blank, mask(text), flags=re.S)
    labels = {label.strip().lower() for label, _ in DEFINITION.findall(body)}
    return [
        (match.start(), match[0])
        for match in REFERENCE.finditer(body)
        if (match[2] or match[1]).strip().lower() not in labels
    ]


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tracked = tracked_files(root)
        self.directories = {
            "/".join(path.split("/")[:end])
            for path in self.tracked or ()
            for end in range(1, path.count("/") + 1)
        }
        version = root / "VERSION"
        self.version = version.read_text().strip() if version.exists() else ""
        self.failures: list[str] = []
        self.external: dict[str, list[str]] = {}
        self.release: list[str] = []  # links to the tag of the current version
        self._anchors: dict[Path, set[str]] = {}

    def fail(self, where: str, message: str) -> None:
        self.failures.append(f"{where}: {message}")

    def exists(self, relative: PurePosixPath) -> bool:
        """Exact-case membership in the checkout: a tracked file or a directory of them."""
        name = relative.as_posix()
        if name == ".":
            return True
        if self.tracked is None:
            return (self.root / name).exists()
        return name in self.tracked or name in self.directories

    def heading(self, where: str, relative: PurePosixPath, fragment: str, link: str) -> None:
        if not fragment or relative.suffix != ".md":
            return
        path = self.root / relative
        if path not in self._anchors:
            self._anchors[path] = anchors(path.read_text())
        if unquote(fragment) not in self._anchors[path]:
            self.fail(where, f"missing heading {link}")

    def local(self, where: str, source: PurePosixPath, link: str) -> None:
        parts = urlsplit(link)
        if parts.path.startswith("/"):  # GitHub resolves these from the repository root
            target = PurePosixPath(unquote(parts.path).lstrip("/"))
        elif parts.path:
            target = source.parent / unquote(parts.path)
        else:
            target = source
        resolved: list[str] = []
        for part in target.parts:
            if part == "..":
                if not resolved:
                    self.fail(where, f"link leaves the repository {link}")
                    return
                resolved.pop()
            elif part != ".":
                resolved.append(part)
        relative = PurePosixPath(*resolved) if resolved else PurePosixPath(".")
        if not self.exists(relative):
            self.fail(where, f"missing local target {link}")
        elif relative.parts[:1] == ("private",) and source.parts[:1] != ("private",):
            self.fail(where, f"public file links outside the public tree {link}")
        else:
            self.heading(where, relative, parts.fragment, link)

    def repository(self, where: str, link: str, match: re.Match) -> None:
        ref, path, fragment = match[1], match[2].rstrip("/"), match[3] or ""
        if ref != REPOSITORY_BRANCH and not ref.startswith("v"):
            self.fail(where, f"the public default branch is {REPOSITORY_BRANCH}: {link}")
            return
        if ref not in (REPOSITORY_BRANCH, f"v{self.version}"):
            self.external.setdefault(link, []).append(where)  # an older release: ask GitHub
            return
        if ref != REPOSITORY_BRANCH:
            self.release.append(where)
        relative = PurePosixPath(unquote(path)) if path else PurePosixPath(".")
        if relative.parts[:1] == ("private",) or not self.exists(relative):
            self.fail(where, f"missing repository target {link}")
        else:
            self.heading(where, relative, fragment, link)

    def check_markdown(self, path: Path) -> None:
        source = PurePosixPath(path.relative_to(self.root).as_posix())
        text = path.read_text()

        def line(offset: int) -> int:
            return text.count("\n", 0, offset) + 1

        for number, row in enumerate(text.splitlines(), 1):
            if re.search(r"\\operatorname\b", row):
                self.fail(f"{source}:{number}", "unsupported math command; use \\mathrm")
            if re.search(r"/libsdca/report/?#", row):
                self.fail(f"{source}:{number}", "link the report as a whole, without an anchor")
        for offset, label in undefined_references(text):
            self.fail(f"{source}:{line(offset)}", f"undefined reference link {label}")
        for offset, link in links(text):
            where = f"{source}:{line(offset)}"
            scheme = urlsplit(link).scheme.lower()
            if scheme in ("mailto", "tel", "data"):
                continue
            if scheme in ("http", "https"):
                match = REPOSITORY.match(link)
                if match:
                    self.repository(where, link, match)
                else:
                    self.external.setdefault(link, []).append(where)
            elif scheme or link.startswith("//"):
                self.fail(where, f"unsupported link scheme {link}")
            elif source.as_posix() in PACKAGE_PAGES and urlsplit(link).path:
                self.fail(where, f"relative link in a guide rendered on a package page {link}")
            else:
                self.local(where, source, link)

    def check_metadata(self, name: str) -> None:
        path = self.root / name
        if not path.exists():
            return
        text = path.read_text()
        for match in BARE.finditer(text):
            link = match[0].rstrip(".,;:!?")
            where = f"{name}:{text.count(chr(10), 0, match.start()) + 1}"
            repository = REPOSITORY.match(link)
            if repository:
                self.repository(where, link, repository)
            else:
                self.external.setdefault(link, []).append(where)

    def run(self, external: bool) -> list[str]:
        for path in markdown_paths(self.root, self.tracked):
            self.check_markdown(path)
        for name in METADATA:
            self.check_metadata(name)
        warnings: list[str] = []
        if not external:
            return warnings
        tag = f"https://github.com/mlapin/libsdca/tree/v{self.version}"
        if self.release and request(tag)[0] == 404:
            warnings.append(
                f"v{self.version} is not on GitHub yet; {len(self.release)} links to it "
                f"({', '.join(self.release[:3])}, ...) resolve once the release is tagged"
            )
        urls = sorted(url for url in self.external if not re.search(r"[${}<>]", url))
        with ThreadPoolExecutor(16) as pool:
            for url, (status, problem) in zip(urls, pool.map(request, urls), strict=True):
                if problem is None:
                    continue
                where = ", ".join(self.external[url][:3])
                if len(self.external[url]) > 3:
                    where += f" (+{len(self.external[url]) - 3} more)"
                if status in UNVERIFIABLE:
                    warnings.append(
                        f"{where}: the host refuses scripted requests ({problem}) {url}"
                    )
                else:
                    self.failures.append(f"{where}: {problem} {url}")
        return warnings


def request(url: str) -> tuple[int | None, str | None]:
    """Fetch `url`; return (status, problem), where problem is None for a working link."""
    address, _, fragment = url.partition("#")
    crate = re.match(r"https://crates\.io/crates/([^/?#]+)/?$", address)
    if crate:  # the crates.io page is a client-side app; ask its API
        address = f"https://crates.io/api/v1/crates/{crate[1]}"
    # GitHub renders heading anchors with a prefix in client-side code; skip those fragments.
    page_anchor = bool(fragment) and urlsplit(address).netloc != "github.com" and not crate
    headers = {"User-Agent": "Mozilla/5.0 (compatible; libsdca-docs-check)", "Accept": "*/*"}
    status: int | None = None
    problem: str | None = None
    for method in ("GET",) if page_anchor else ("HEAD", "GET"):
        request_ = urllib.request.Request(address, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request_, timeout=30) as response:
                status = response.status
                body = response.read(4_000_000) if method == "GET" else b""
        except urllib.error.HTTPError as error:  # some servers refuse HEAD: retry with GET
            status, problem = error.code, f"HTTP {error.code}"
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            status, problem = None, f"unreachable ({getattr(error, 'reason', error)})"
            continue
        if page_anchor:
            name = re.escape(unquote(fragment))
            page = body.decode("utf-8", "replace")
            if not re.search(rf"""\b(?:id|name)\s*=\s*["']{name}["']""", page):
                return status, "missing anchor"
        return status, None
    return status, problem


def check_docs(root: Path = ROOT, external: bool = False) -> None:
    checker = Checker(root)
    for warning in checker.run(external):
        print(f"warning: {warning}", file=sys.stderr)
    if checker.failures:
        failures = "\n".join(sorted(set(checker.failures)))
        raise SystemExit("Documentation check failed:\n" + failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--external", action="store_true", help="also request external URLs")
    external = parser.parse_args().external
    check_docs(external=external)
    print("Documentation links and math: OK" + (" (with external URLs)" if external else ""))


if __name__ == "__main__":
    main()

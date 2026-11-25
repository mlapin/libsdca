"""One GPU test session from a git archive (on any machine with an NVIDIA GPU), in one command:

    tar xzf ARCHIVE.tgz tools/gpu_session.py && python3 tools/gpu_session.py ARCHIVE.tgz

Extracts the archive into OUTPUT/src, makes OUTPUT/venv with the CUDA test dependencies
(pyproject's test-torch, test-jax and jax-cuda extras, cmake and ninja), builds the native
library (python/build.py) and runs these steps, each past the failure of another:

- pytest of the core, torch and jax groups under -n: the CUDA tests (CUDA devices of
  libsdca.torch, the GPU backends of libsdca.jax and of the conformance matrix) run where
  a GPU is visible and skip otherwise; conformance records go to OUTPUT/conformance
  (SDCA_CONFORMANCE_RECORD), from which tests/python/conformance/manifest.py writes a
  reviewed GPU manifest;
- the serial tests without -n, and on its own each registered marker whose description
  says CUDA (pyproject's cuda: the rule-table and cross-interface CUDA rows);
- tests/python/release_numerics.py --device cuda and tools/perf_gate.py;
- ctest (BUILD_TESTING, the C ABI library, warnings as errors) and cargo test (rustup
  installs a toolchain into OUTPUT where cargo is missing).

OUTPUT (default ./gpu-session-<commit>) receives env.json, logs/<step>.log,
junit/<step>.xml, summary.json and summary.md, packed into OUTPUT.tgz. Exit 1 when a step
failed. --dry-run collects the tests and configures the builds without running them (no
GPU needed); --python runs an existing interpreter's packages, installing nothing.
Standard library only: it runs before the virtual environment exists.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

STEPS = ("core", "torch", "jax", "serial", "markers", "numerics", "perf", "ctest", "cargo")
EXTRAS = ("test-torch", "test-jax", "jax-cuda")
TOOLS = ("cmake>=3.21", "ninja")
# A registered marker whose description says CUDA (pyproject's cuda) runs on its own too.
CUDA_MARKER = re.compile(r"^@pytest\.mark\.(\w+):.*\bCUDA\b", re.MULTILINE)
# jax[cuda13] needs an NVIDIA driver 580 or newer; older drivers take jax[cuda12].
CUDA13_DRIVER = 580
APT = ("build-essential", "python3-dev", "python3-venv", "libblas-dev", "pkg-config", "curl")
CUDA_NAME = re.compile(r"cuda|gpu", re.IGNORECASE)


@dataclass
class Step:
    name: str
    command: list[str]
    status: str = "not run"  # passed, failed, skipped
    exit_code: int | None = None
    seconds: float = 0.0
    tests: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    cuda_tests: int = 0
    note: str = ""


class Session:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = args.output.resolve()
        self.src: Path = args.src
        self.python = ""
        self.steps: list[Step] = []
        self.env = dict(os.environ)
        self.env.update(
            PYTHONPATH=str(self.src / "python"),
            SDCA_CONFORMANCE_RECORD=str(self.out / "conformance"),
            SDCA_CACHE_DIR=str(self.out / "cache" / "libsdca"),
            JAX_COMPILATION_CACHE_DIR=str(self.out / "cache" / "jax"),
            # Each pytest worker is a JAX process: without this each takes 75% of the GPU.
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            PYTHONUNBUFFERED="1",
        )

    def use(self, python: str) -> None:
        """Run python, with its bin directory (cmake, ninja, ctest of the venv) first on PATH."""
        self.python = python
        self.env["PATH"] = f"{Path(python).parent}{os.pathsep}{self.env.get('PATH', '')}"
        for name in ("logs", "junit", "conformance"):
            (self.out / name).mkdir(parents=True, exist_ok=True)

    def run(self, name: str, command: list[str], cwd: Path | None = None) -> Step:
        """A step's command, its output in logs/<name>.log; a failure does not stop the session."""
        step = Step(name, command)
        log = self.out / "logs" / f"{name}.log"
        print(f"== {name}: {' '.join(command)}", flush=True)
        start = time.monotonic()
        with log.open("w") as f:
            f.write(f"$ {' '.join(command)}\n")
            f.flush()
            done = subprocess.run(
                command, cwd=cwd or self.src, env=self.env, stdout=f, stderr=subprocess.STDOUT
            )
        step.seconds = round(time.monotonic() - start, 1)
        step.exit_code = done.returncode
        step.status = "passed" if done.returncode == 0 else "failed"
        print(f"   exit {step.exit_code} in {step.seconds} s, log {log}", flush=True)
        self.steps.append(step)
        return step

    def skip(self, name: str, note: str) -> None:
        print(f"== {name}: skipped ({note})", flush=True)
        self.steps.append(Step(name, [], status="skipped", note=note))

    def output(self, command: list[str]) -> str:
        done = subprocess.run(
            command, cwd=self.src, env=self.env, capture_output=True, text=True, check=False
        )
        return done.stdout + done.stderr


def extract(source: Path, out: Path) -> tuple[Path, str]:
    """The source tree of a git archive (into out/src) or a directory, and its commit."""
    if source.is_dir():
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        return source.resolve(), commit or "unknown"
    src = out / "src"
    with tarfile.open(source) as archive:
        commit = str(archive.pax_headers.get("comment", "unknown"))
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)
        try:
            archive.extractall(src, filter="data")
        except TypeError:  # Python without extraction filters (before 3.10.12, 3.11.4)
            archive.extractall(src)
    return src, commit


def driver_major() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    done = subprocess.run(query, capture_output=True, text=True)
    match = re.match(r"(\d+)", done.stdout.strip())
    return int(match.group(1)) if match else None


def extras(src: Path, names: tuple[str, ...]) -> list[str]:
    """The requirements of pyproject.toml's optional dependencies names (one line each)."""
    text = (src / "pyproject.toml").read_text()
    section = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    found: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^([\w-]+)\s*=\s*(\[.*\])\s*$", line)
        if match and match.group(1) in names:
            found += [r for r in ast.literal_eval(match.group(2)) if r not in found]
    return found


def install(s: Session) -> None:
    """apt packages (as root, where missing), the virtual environment and its requirements."""
    if os.geteuid() == 0 and shutil.which("apt-get") and not s.args.no_apt:
        missing = [
            p for p in APT if subprocess.run(["dpkg", "-s", p], capture_output=True).returncode
        ]
        if missing:
            s.run("apt", ["sh", "-c", "apt-get update && apt-get install -y " + " ".join(missing)])
    s.run("venv", [sys.executable, "-m", "venv", str(s.out / "venv")])
    pip = [s.python, "-m", "pip", "install", "--progress-bar", "off"]
    s.run("pip", [*pip, "--upgrade", "pip"])
    requirements = extras(s.src, EXTRAS) + list(TOOLS)
    driver = driver_major()
    jax = s.args.jax or (
        "jax[cuda12]>=0.7.2" if driver is not None and driver < CUDA13_DRIVER else None
    )
    if jax:
        requirements = [r for r in requirements if not r.startswith("jax[cuda")] + [jax]
    if s.args.torch_index:
        torch = [r for r in requirements if re.match(r"torch\b", r)]
        s.run("pip-torch", [*pip, "--index-url", s.args.torch_index, *torch])
    s.run("pip-requirements", [*pip, *requirements])


def environment(s: Session, commit: str) -> dict[str, object]:
    probe = (
        "import json, platform, sys\n"
        "v = {'python': sys.version.split()[0], 'platform': platform.platform()}\n"
        "for m in ('numpy', 'torch', 'jax', 'jaxlib', 'triton', 'jax_triton', 'optax'):\n"
        "    try:\n"
        "        v[m] = __import__(m).__version__\n"
        "    except Exception as e:\n"
        "        v[m] = f'unavailable: {type(e).__name__}'\n"
        "try:\n"
        "    import torch\n"
        "    v['torch_cuda'] = torch.version.cuda\n"
        "    v['torch_devices'] = [torch.cuda.get_device_name(i)\n"
        "                          for i in range(torch.cuda.device_count())]\n"
        "except Exception as e:\n"
        "    v['torch_devices'] = f'unavailable: {e}'\n"
        "try:\n"
        "    import jax\n"
        "    v['jax_devices'] = [str(d) for d in jax.devices()]\n"
        "except Exception as e:\n"
        "    v['jax_devices'] = f'unavailable: {e}'\n"
        "print(json.dumps(v))\n"
    )
    text = s.output([s.python, "-c", probe]).strip().splitlines()
    try:
        versions = json.loads(text[-1]) if text else {}
    except json.JSONDecodeError:
        versions = {"probe": text[-20:]}
    gpu = "none"
    if shutil.which("nvidia-smi"):
        query = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        gpu = s.output(query).strip()
    env: dict[str, object] = {
        "commit": commit,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": gpu,
        "nproc": os.cpu_count(),
        "dry_run": s.args.dry_run,
        "versions": versions,
    }
    (s.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    return env


def pytest(s: Session, name: str, *selection: str, parallel: bool = True) -> Step:
    command = [s.python, "-m", "pytest", "tests/python", *selection, "-p", "no:cacheprovider"]
    if parallel and not s.args.dry_run:
        command += ["-n", str(s.args.workers), "--dist", "worksteal"]
    if s.args.dry_run:
        command += ["--collect-only", "-q"]
    else:
        command += ["-rfE", f"--junitxml={s.out / 'junit' / f'{name}.xml'}"]
    step = s.run(name, command + s.args.pytest_args)
    if step.exit_code == 5:
        step.status, step.note = "skipped", "no tests selected"
    if s.args.dry_run:
        collected(s, step)
    else:
        junit(s.out / "junit" / f"{name}.xml", step)
    return step


def collected(s: Session, step: Step) -> None:
    """A dry run's counts: the collected test ids, those of CUDA devices or backends."""
    lines = (s.out / "logs" / f"{step.name}.log").read_text().splitlines()
    ids = [line for line in lines if "::" in line]
    step.tests = {"collected": len(ids)}
    step.cuda_tests = sum(1 for i in ids if CUDA_NAME.search(i))


def junit(path: Path, step: Step) -> None:
    """Counts, failing ids and the CUDA tests run (not skipped) of a JUnit file."""
    if not path.exists():
        step.note = "no JUnit report"
        return
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in ET.parse(path).getroot().iter("testcase"):
        totals["tests"] += 1
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        outcome = next((c.tag for c in case if c.tag in ("failure", "error", "skipped")), None)
        if outcome == "skipped":
            totals["skipped"] += 1
        elif outcome is not None:
            totals["failures" if outcome == "failure" else "errors"] += 1
            step.failures.append(test_id)
        if outcome != "skipped" and CUDA_NAME.search(test_id):
            step.cuda_tests += 1
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    step.tests = totals


def markers(s: Session) -> None:
    listed = s.output([s.python, "-m", "pytest", "--markers", "-p", "no:cacheprovider"])
    registered = CUDA_MARKER.findall(listed)
    if not registered:
        s.skip("markers", "no registered marker mentions CUDA")
    for marker in registered:
        pytest(s, f"marker-{marker}", "--group", "all", "-m", marker)


def ctest(s: Session) -> None:
    build = s.out / "build" / "cpp"
    configure = ["cmake", "-S", str(s.src), "-B", str(build), "-G", "Ninja", "-DBUILD_TESTING=ON"]
    configure += ["-DSDCA_BUILD_C_API=ON", "-DSDCA_WARNINGS_AS_ERRORS=ON"]
    if s.run("cmake-configure", configure).status != "passed":
        return
    if s.args.dry_run:
        s.run("ctest", ["ctest", "--test-dir", str(build), "-N"])
        return
    if (
        s.run("cmake-build", ["cmake", "--build", str(build), "-j", str(os.cpu_count())]).status
        != "passed"
    ):
        return
    report = s.out / "junit" / "ctest.xml"
    command = ["ctest", "--test-dir", str(build), "--output-on-failure", "--parallel", "8"]
    junit(report, s.run("ctest", [*command, "--output-junit", str(report)]))


def cargo(s: Session) -> None:
    manifest = str(s.src / "rust" / "Cargo.toml")
    s.env["CARGO_TARGET_DIR"] = str(s.out / "build" / "cargo")
    if shutil.which("cargo", path=s.env.get("PATH")) is None:
        home = s.out / "rust"
        s.env.update(CARGO_HOME=str(home / "cargo"), RUSTUP_HOME=str(home / "rustup"))
        s.env["PATH"] = f"{home / 'cargo' / 'bin'}{os.pathsep}{s.env.get('PATH', '')}"
        if shutil.which("cargo", path=s.env["PATH"]) is None:
            if s.args.dry_run:
                s.skip("cargo", "cargo is not installed (dry run)")
                return
            rustup = (
                "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path"
            )
            if s.run("rustup", ["sh", "-c", rustup]).status != "passed":
                return
    if s.args.dry_run:
        s.run(
            "cargo",
            [
                "cargo",
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                manifest,
            ],
        )
        return
    step = s.run("cargo", ["cargo", "test", "--manifest-path", manifest])
    text = (s.out / "logs" / "cargo.log").read_text()
    counts = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    passed, failed, ignored = (sum(int(c[i]) for c in counts) for i in range(3))
    step.tests = {"passed": passed, "failures": failed, "skipped": ignored}


def summarize(s: Session, env: dict[str, object]) -> int:
    failed = [step.name for step in s.steps if step.status == "failed"]
    summary = {**env, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "failed_steps": failed}
    summary["steps"] = [asdict(step) for step in s.steps]
    (s.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# GPU session {env['commit']}",
        "",
        f"- GPU: {env['gpu']}",
        f"- Versions: {json.dumps(env['versions'])}",
        f"- Dry run: {s.args.dry_run}",
        "",
        "| Step | Status | Seconds | Tests | CUDA tests | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for step in s.steps:
        tests = ", ".join(f"{k} {v}" for k, v in step.tests.items())
        lines.append(
            f"| {step.name} | {step.status} | {step.seconds} | {tests} | {step.cuda_tests} "
            f"| {step.note} |"
        )
    for step in s.steps:
        if step.failures:
            lines += ["", f"## {step.name}: {len(step.failures)} failing", ""]
            lines += [f"- `{f}`" for f in step.failures[:200]]
    (s.out / "summary.md").write_text("\n".join(lines) + "\n")
    with tarfile.open(f"{s.out}.tgz", "w:gz") as archive:
        for name in ("env.json", "summary.json", "summary.md", "logs", "junit", "conformance"):
            if (s.out / name).exists():
                archive.add(s.out / name, arcname=f"{s.out.name}/{name}")
        for name in ("release_numerics", "perf_gate.json"):
            if (s.out / name).exists():
                archive.add(s.out / name, arcname=f"{s.out.name}/{name}")
    print(f"summary: {s.out / 'summary.md'}; results: {s.out}.tgz", flush=True)
    print("failed steps: " + (", ".join(failed) or "none"), flush=True)
    return 1 if failed else 0


def parse() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=root,
        help="git archive tarball or source tree (default: this checkout)",
    )
    parser.add_argument("--output", type=Path, help="results directory")
    parser.add_argument("--python", help="an interpreter with the dependencies (installs nothing)")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="pytest -n (default: CPUs, at most 8)",
    )
    parser.add_argument("--steps", default=",".join(STEPS), help=f"of {','.join(STEPS)}")
    parser.add_argument("--dry-run", action="store_true", help="collect and configure only")
    parser.add_argument("--torch-index", help="pip index of a CUDA build of PyTorch")
    parser.add_argument("--jax", help="the JAX requirement (default: the jax-cuda extra's)")
    parser.add_argument("--no-apt", action="store_true", help="never run apt-get")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="passed to every pytest step (last)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse()
    unknown = set(args.steps.split(",")) - set(STEPS)
    if unknown:
        raise SystemExit(f"unknown steps: {', '.join(sorted(unknown))}")
    stem = args.source.name.removesuffix(".tgz").removesuffix(".tar.gz").removeprefix("rw-")
    args.output = args.output or Path(f"gpu-session-{stem}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.src, commit = extract(args.source, args.output.resolve())
    s = Session(args)
    if args.python:
        s.use(args.python)
    elif args.dry_run:
        s.use(sys.executable)
    else:
        s.use(str(s.out / "venv" / "bin" / "python"))
        install(s)
    env = environment(s, commit)
    if s.run("build", [s.python, "python/build.py"]).status != "passed":
        return summarize(s, env)
    steps = set(args.steps.split(","))
    for group in ("core", "torch", "jax"):
        if group in steps:
            pytest(s, group, "--group", group)
    if "serial" in steps:
        pytest(s, "serial", "--group", "all", "-m", "serial", parallel=False)
    if "markers" in steps:
        markers(s)
    for name, command in (
        (
            "numerics",
            [
                "tests/python/release_numerics.py",
                "--device",
                "cuda",
                "--output",
                str(s.out / "release_numerics"),
            ],
        ),
        ("perf", ["tools/perf_gate.py", "--output", str(s.out / "perf_gate.json")]),
    ):
        if name in steps:
            if args.dry_run:
                s.run(name, [s.python, command[0], "--help"])
            else:
                s.run(name, [s.python, *command])
    if "ctest" in steps:
        ctest(s)
    if "cargo" in steps:
        cargo(s)
    return summarize(s, env)


if __name__ == "__main__":
    sys.exit(main())

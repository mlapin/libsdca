"""Run literal README examples after one install in each fresh uv/conda environment.

Example: python tools/readme_smoke.py --wheel dist/libsdca-*.whl --output build/readme-smoke
CPU: --profiles numpy sklearn torch jax. Linux CUDA: --device cuda --profiles torch jax-cuda.
The environment creator must already exist. No test dependencies or repair installs are
allowed in the environment. A local release wheel substitutes for the unpublished package;
omit --wheel to test exactly the current package-index installation instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "numpy": "",
    "sklearn": "sklearn",
    "torch": "torch",
    "jax": "jax",
    "jax-cuda": "jax-cuda",
}


def snippet(readme: str, name: str) -> str:
    match = re.search(
        r"<!-- smoke: " + re.escape(name) + r" -->\s*```\w+\n(.*?)\n```", readme, re.S
    )
    if match is None:
        raise ValueError(f"Missing runnable README example: {name}")
    return match[1] + "\n"


def command(readme: str, profile: str) -> list[str]:
    requirement = "libsdca" + (f"[{PROFILES[profile]}]" if PROFILES[profile] else "")
    for text in re.findall(r"`(pip install [^`]+)`", readme):
        words = shlex.split(text)
        if words == ["pip", "install", requirement]:
            return words
    raise ValueError(f"Missing one-command install for {profile}")


def clean_environment() -> dict[str, str]:
    prefixes = (
        "SDCA_",
        "JAX_",
        "XLA_",
        "PYTHON",
        "CONDA_",
        "VIRTUAL_ENV",
        "TORCH_EXTENSIONS",
        "TRITON_",
    )
    return {
        k: v for k, v in os.environ.items() if k == "CONDA_PKGS_DIRS" or not k.startswith(prefixes)
    }


def documented_environment(readme: str, manager: str) -> list[str]:
    """The README command that creates the environment: the first line of its block (the
    lines after it activate the environment and install the package)."""
    return shlex.split(snippet(readme, f"env-{manager}").splitlines()[0])


def environment_command(readme: str, manager: str, python: str, path: Path) -> list[str]:
    """Only relocate the literal README environment; alternate Python is an explicit override."""
    command = documented_environment(readme, manager)
    replacements = {".venv": str(path), "3.12": python, "python=3.12": f"python={python}"}
    return [replacements.get(word, word) for word in command]


def run(command: list[str], directory: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a") as stream:
        stream.write("+ " + shlex.join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=directory,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1200,
        )


def verify_code(profile: str, device: str) -> str:
    base = (
        "\nimport pathlib, libsdca\nprint('installed:', pathlib.Path(libsdca.__file__).resolve())\n"
    )
    if profile == "torch":
        fused = (
            "assert any(name in ce.grad_fn.name() for name in "
            "('_FusedWeightedEntropyBackward', 'CudaWeightedEntropyFunction')), ce.grad_fn.name()\n"
            if device == "cuda"
            else ""
        )
        paths = "('kernel',)" if device == "cuda" else "('kernel', 'native')"
        explained = (
            "from libsdca.torch import explain_backend\n"
            "_path = explain_backend(TopKHingeLoss(k=2), logits, targets)\n"
            f"assert _path[0] in {paths}, _path\n"
        )
        return (
            base
            + "assert ce.isfinite() and hinge.isfinite()\n"
            + f"assert logits.grad.isfinite().all()\nassert logits.device.type == {device!r}\n"
            + fused
            + explained
        )
    if profile.startswith("jax"):
        check = (
            base
            + "assert jnp.isfinite(value).all() and jnp.isfinite(gradient).all()\ngradient.block_until_ready()\n"
        )
        check += f"assert jax.default_backend() == {('gpu' if device == 'cuda' else 'cpu')!r}\n"
        if device == "cuda":
            check += "from libsdca import _jax_cuda\nassert _jax_cuda.status()[0], _jax_cuda.status()[1]\n"
            check += (
                "_lowered = jax.jit(jax.value_and_grad(cross_entropy)).lower(logits, targets)\n"
                "assert _jax_cuda.TARGET in _lowered.as_text()\n"
            )
            check += (
                "_z = jnp.arange(28, dtype=jnp.float32).reshape(4, 7) / 100\n"
                "_y = jnp.arange(4) % 7\n"
                "_hinge = jax.jit(jax.value_and_grad(lambda z: topk_hinge_loss(z, _y, k=2)))\n"
                "assert _jax_cuda.TARGET in _hinge.lower(_z).as_text(), 'top-2 hinge bypassed fused CUDA'\n"
                "_value, _gradient = _hinge(_z)\n"
                "_gradient.block_until_ready()\n"
                "assert jnp.isfinite(_value).all() and jnp.isfinite(_gradient).all()\n"
            )
        return check
    if profile == "numpy":
        return base + "assert model.status == 'solved' and np.isfinite(model.weights).all()\n"
    return base + "assert clf.predict(X[:5]).shape == (5,)\n"


def provenance_code(wheel: Path | None) -> str:
    if wheel is None:
        return ""
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return (
        "\nimport importlib.metadata, json\n"
        "origin = json.loads(importlib.metadata.distribution('libsdca').read_text('direct_url.json'))\n"
        f"assert origin['archive_info']['hashes']['sha256'] == {digest!r}, origin\n"
        "print('wheel sha256:', origin['archive_info']['hashes']['sha256'])\n"
    )


def validate_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.require_fused_cpu and (args.device != "cpu" or args.without_compiler):
        parser.error("--require-fused-cpu needs --device cpu and an available compiler")
    if args.device == "cuda":
        if sys.platform != "linux":
            parser.error("CUDA README tests require Linux; CUDA on macOS is unsupported")
        if set(args.profiles) - {"torch", "jax-cuda"}:
            parser.error("CUDA profiles are torch and jax-cuda; the other interfaces are CPU APIs")
    elif "jax-cuda" in args.profiles:
        parser.error("jax-cuda requires --device cuda")
    if args.wheel and sys.platform == "linux":
        with zipfile.ZipFile(args.wheel) as archive:
            if not any(
                "/_jax_cuda_native" in name and name.endswith(".so") for name in archive.namelist()
            ):
                parser.error("the Linux release wheel must contain the bundled JAX CUDA handler")


def run_examples(
    profile: str,
    python: str,
    directory: Path,
    env: dict[str, str],
    log: Path,
    readme: str,
    require_fused_cpu: bool = False,
    device: str = "cpu",
) -> dict[str, int] | None:
    run([python, str(directory / "example.py")], directory, env, log)
    contracts = directory / "contracts.py"
    shutil.copyfile(ROOT / "tools/readme_contract_smoke.py", contracts)
    shutil.copyfile(ROOT / "tests/python/hinge_order_cases.py", directory / "hinge_order_cases.py")
    run([python, str(contracts), profile, "--device", device], directory, env, log)
    if require_fused_cpu and profile in ("torch", "jax"):
        probe = directory / "cpu-dispatch.py"
        shutil.copyfile(ROOT / "tools/readme_cpu_smoke.py", probe)
        run(
            [python, str(probe), profile, *(["--require-jax-only"] if profile == "jax" else [])],
            directory,
            env,
            log,
        )
    if device == "cuda":
        probe = directory / "cuda-dispatch.py"
        shutil.copyfile(ROOT / "tools/readme_cuda_smoke.py", probe)
        run([python, str(probe), profile], directory, env, log)
    if profile in ("numpy", "sklearn"):
        probe = directory / "precision.py"
        shutil.copyfile(ROOT / "tools/readme_precision_smoke.py", probe)
        run(
            [python, str(probe), *(["--sklearn"] if profile == "sklearn" else [])],
            directory,
            env,
            log,
        )
        rows = json.loads((directory / "precision.json").read_text())
        return {
            "passed": sum(row["status"] == "PASS" for row in rows),
            "known_fp64_exceptions": sum(row["status"] == "KNOWN_FP64_EXCEPTION" for row in rows),
        }
    return None


def check_profile(args: argparse.Namespace, profile: str, readme: str) -> dict[str, object]:
    directory = args.output / profile
    directory.mkdir(parents=True, exist_ok=args.phase == "run")
    environment = directory / "env"
    log = directory / "commands.log"
    env = clean_environment()
    # Give this test its own caches so first-use compilation cannot be inherited.
    env.update(
        SDCA_CACHE_DIR=str(directory / "sdca-cache"),
        TORCH_EXTENSIONS_DIR=str(directory / "torch-cache"),
        TRITON_CACHE_DIR=str(directory / "triton-cache"),
        PIP_CACHE_DIR=str(args.output.parent / "download-cache"),
    )
    # Downloaded wheels may be reused; installed packages and compiled kernels may not.
    env.pop("PIP_NO_CACHE_DIR", None)
    create = environment_command(readme, args.manager, args.python, environment)
    install = command(readme, profile)
    python = str(environment / "bin/python")
    actual = [python, "-m", *install]
    if args.wheel:
        # Only substitute the candidate artifact, never dependencies or package options.
        digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
        actual[-1] += " @ " + args.wheel.as_uri() + "#sha256=" + digest
    name = "jax" if profile == "jax-cuda" else profile
    example = snippet(readme, name)
    script = directory / "example.py"
    script.write_text(example + verify_code(profile, args.device) + provenance_code(args.wheel))
    record: dict[str, object] = {
        "profile": profile,
        "device": args.device,
        "manager": args.manager,
        "without_compiler": args.without_compiler,
        "require_fused_cpu": args.require_fused_cpu,
        "started_cet": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "readme_sha256": hashlib.sha256(readme.encode()).hexdigest(),
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest() if args.wheel else None,
        "documented_install": install,
        "documented_environment": documented_environment(readme, args.manager),
        "environment_create": create,
        "python_override": args.python if args.python != "3.12" else None,
        "install": actual,
        "example": example,
        "status": "FAIL",
    }
    started = time.monotonic()
    try:
        if args.phase != "run":
            run(create, directory, env, log)
            run(
                [
                    python,
                    "-c",
                    "import importlib.util,json,pathlib; "
                    "names=('numpy','torch','jax','sklearn','libsdca'); "
                    "found={n:importlib.util.find_spec(n) is not None for n in names}; "
                    "pathlib.Path('before-install.json').write_text(json.dumps(found)); "
                    "assert not any(found.values()), found",
                ],
                directory,
                env,
                log,
            )
            run(actual, directory, env, log)
        run([python, "-m", "pip", "check"], directory, env, log)
        runtime_env = dict(env)
        if args.without_compiler:
            tool_path = directory / "runtime-tools"
            tool_path.mkdir(exist_ok=True)
            for tool in ("sh", "bash", "ld", "as", "uname"):
                found = shutil.which(tool)
                link = tool_path / tool
                if found and not link.exists():
                    link.symlink_to(found)
            runtime_env.pop("CC", None)
            runtime_env.pop("CXX", None)
            runtime_env["PATH"] = str(environment / "bin") + os.pathsep + str(tool_path)
        if args.phase != "install":
            record["native_precision"] = run_examples(
                profile,
                python,
                directory,
                runtime_env,
                log,
                readme,
                args.require_fused_cpu,
                args.device,
            )
            record["numerical_contracts"] = json.loads((directory / "contracts.json").read_text())
        freeze = subprocess.check_output([python, "-m", "pip", "freeze"], env=env, text=True)
        (directory / "packages.txt").write_text(freeze)
        record["status"] = "INSTALLED" if args.phase == "install" else "PASS"
    except (subprocess.SubprocessError, OSError) as error:
        record["error"] = str(error)
    if args.cleanup and record["status"] == "PASS":
        for path in (
            environment,
            *(directory / name for name in ("sdca-cache", "torch-cache", "triton-cache")),
        ):
            shutil.rmtree(path, ignore_errors=True)
        record["environment_removed"] = True
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "install", "run"),
        default="all",
        help="split package download from exclusive-device example execution",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove environments and runtime caches after a passing execution; keep all records",
    )
    parser.add_argument(
        "--require-fused-cpu",
        action="store_true",
        help="also execute and verify shared Torch/JAX CPU distillation",
    )
    parser.add_argument("--manager", choices=("uv", "conda"), default="uv")
    parser.add_argument("--python", default="3.12", help="override the README's Python version")
    parser.add_argument(
        "--without-compiler",
        action="store_true",
        help="remove host compilers from PATH; checks prebuilt/fallback install paths",
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=["numpy", "sklearn", "torch", "jax"]
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.wheel:
        args.wheel = args.wheel.resolve()
    validate_options(parser, args)
    args.output.mkdir(parents=True, exist_ok=True)
    readme = (ROOT / "README.md").read_text()
    records = []
    for profile in args.profiles:
        result = check_profile(args, profile, readme)
        records.append(result)
        print(profile, result["status"], result["seconds"], flush=True)
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return int(any(record["status"] == "FAIL" for record in records))


if __name__ == "__main__":
    sys.exit(main())

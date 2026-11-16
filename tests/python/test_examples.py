"""The Python examples run to completion (each checks its own results).

torch_mnist_lenet.py downloads MNIST and is not run here.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "python"
# The sdist ships the tests without the examples.
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="no examples next to the tests")
NEEDS = {
    "numpy_fit.py": "numpy",
    "sklearn_classifier.py": "sklearn",
    "jax_losses.py": "jax",
    "torch_drop_in_cross_entropy.py": "torch",
    "torch_losses_tour.py": "torch",
    "torch_distillation.py": "torch",
}


def test_every_example_is_listed():
    assert {p.name for p in EXAMPLES.glob("*.py")} == {*NEEDS, "torch_mnist_lenet.py"}


@pytest.mark.parametrize("script", sorted(NEEDS))
def test_example_runs(script: str):
    if importlib.util.find_spec(NEEDS[script]) is None:
        pytest.skip(f"{script} needs {NEEDS[script]}")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]

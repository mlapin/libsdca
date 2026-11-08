import importlib.util
import os
import sys

import pytest
from child_process import run_python


def _buildable() -> bool:
    """Whether the CPU extension can build here (imports torch: call only where it exists)."""
    from libsdca import _cpu_entropy

    return _cpu_entropy.buildable()


# 100 MiB tensors sit above glibc's dynamic mmap ceiling (32 MiB): each one faults its pages in.
CHILD = """
import resource, torch
import libsdca._cpu_entropy as ext
from libsdca.torch import cross_entropy

def faults():
    start = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    for _ in range(8):
        torch.empty(25 << 20).fill_(1.0)
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt - start

before = faults()
z = torch.randn(64, 1000, requires_grad=True)
cross_entropy(z, torch.randint(0, 1000, (64,))).backward()
assert ext.operators.cache_info().currsize == 1
after = faults()
assert after >= 0.9 * before, (before, after)
"""


@pytest.mark.skipif(
    not (sys.platform == "linux" and importlib.util.find_spec("torch") and _buildable()),
    reason="glibc, torch and the tools of the CPU extension",
)
class CPUExtensionProcessStateTests:
    @pytest.mark.slow
    def test_extension_leaves_the_allocator_alone(self):
        """The CPU extension must not retune malloc for the rest of the process."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MALLOC_", "GLIBC_"))}
        env["SDCA_CPU_EXTENSION"] = "on"
        result = run_python("-c", CHILD, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-2000:]

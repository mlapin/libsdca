"""Native calls keep subnormals under a caller's flush-to-zero and restore it."""

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from libsdca import coordinate_update, loss_and_grad


def _flushes() -> bool:
    return float(torch.tensor(1e-300, dtype=torch.float64) * 1e-10) == 0.0


class FlushToZeroTests:
    @pytest.fixture(autouse=True)
    def _flush_to_zero(self) -> Iterator[None]:
        if not torch.set_flush_denormal(True):
            pytest.skip("torch cannot set flush-to-zero on this CPU")
        try:
            assert _flushes()
            yield
        finally:
            torch.set_flush_denormal(False)

    def test_subnormal_gradients_on_any_thread_count(self):
        # exp(-740) is subnormal; compare bits, as flush-to-zero also zeroes comparisons.
        z = np.tile([0.0, -740.0], (50000, 1))
        tiny = np.array([4.2e-322]).view(np.uint64)[0]
        for threads in (1, 3):
            _, gradients = loss_and_grad(z, np.zeros(len(z), int), num_threads=threads)
            assert _flushes()
            assert np.all(gradients[:, 1].view(np.uint64) == tiny)

    def test_proximal_entropy_converges_and_errors_restore_the_caller(self):
        z, mask = np.array([[0.0, 20.0, 20.0, -100.0]]), np.array([[True, False, True, False]])
        values, _ = loss_and_grad(z, mask, "proximal_cross_entropy", alpha=1e300)
        assert _flushes()
        assert round(abs(values[0] - np.log(2.0)), 12) == 0
        with pytest.raises(ValueError, match="exactly one target"):
            loss_and_grad(z, mask, "topk_hinge_alpha")
        assert _flushes()
        coordinate_update(z, [0], "cross_entropy", C=2.0)
        assert _flushes()

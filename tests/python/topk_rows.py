"""Top-k cross-entropy rows whose cap set is known, with 60-digit closed-form references.

Rows [0, -m, -m - 40, ..., -m - 40] of label 0 cap their largest competitor at 1 / k; the
others share (k - 1) / k equally. The odds are o = -m + (k - 1)(log((C - 2) / (k - 1)) - 40)
/ k + log k; the loss is softplus(o), the target entry -sigmoid(o), the capped entry
sigmoid(o) / k and every free entry sigmoid(o) (k - 1) / (k (C - 2)). The gaps reach odds of
-689, normal in float64.
"""

import mpmath as mp
import numpy as np
import numpy.typing as npt
from mpmath_types import MPReal

CAPPED_GAPS = [-30 + 7.25 * i for i in range(10)] + [40 + 13.25 * i for i in range(48)]


def capped_rows(classes: int) -> npt.NDArray[np.float64]:
    """One float64 row per gap m of CAPPED_GAPS."""
    z = np.zeros((len(CAPPED_GAPS), classes))
    z[:, 1:] = -np.array(CAPPED_GAPS)[:, None]
    z[:, 2:] -= 40
    return z


def capped_reference(gap: float, classes: int, k: int) -> tuple[MPReal, MPReal, MPReal, MPReal]:
    """Loss, target entry, capped entry and free entry of the row at gap (1 < k < C - 1)."""
    with mp.workdps(60):
        spread = mp.log(mp.mpf(classes - 2) / (k - 1)) - 40
        odds = -mp.mpf(gap) + (k - 1) * spread / k + mp.log(k)
        mass = 1 / (1 + mp.exp(-odds))
        return mp.log1p(mp.exp(odds)), -mass, mass / k, mass * (k - 1) / (k * (classes - 2))

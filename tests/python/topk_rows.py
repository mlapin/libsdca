"""Top-k cross-entropy rows whose cap set is known: [0, -m, -m - 40, ..., -m - 40] of label 0
cap their largest competitor at 1 / k; the others share (k - 1) / k equally. The gaps reach
odds of -689, normal in float64 (references: conformance.references)."""

import numpy as np
import numpy.typing as npt

CAPPED_GAPS = [-30 + 7.25 * i for i in range(10)] + [40 + 13.25 * i for i in range(48)]


def capped_rows(classes: int) -> npt.NDArray[np.float64]:
    """One float64 row per gap m of CAPPED_GAPS."""
    z = np.zeros((len(CAPPED_GAPS), classes))
    z[:, 1:] = -np.array(CAPPED_GAPS)[:, None]
    z[:, 2:] -= 40
    return z

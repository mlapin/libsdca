"""Distinct scores must stay ordered when their unit margins round to ties."""

import numpy as np


def rows(classes: int, power: int):
    # Move the label and winner across row/chunk boundaries. The last row is a true tie,
    # where the first competitor remains the documented deterministic subgradient.
    z = np.full((4, classes), -100.0)
    labels = np.array([0, classes - 1, 1, 0], dtype=np.int64)
    expected = np.zeros_like(z)
    for row, label in enumerate(labels):
        competitors = [j for j in range(classes) if j != label]
        low, high = competitors[0], competitors[-1]
        z[row, label] = 0
        z[row, low] = 2.0**power
        z[row, high] = 2.0 ** (power + (row != 3))
        expected[row, low if row == 3 else high] = 1
        expected[row, label] = -1
    return z, labels, expected

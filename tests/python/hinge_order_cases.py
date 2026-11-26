"""Distinct scores must stay ordered when their unit margins round to ties."""

import numpy as np


def rows(classes: int, power: int):
    # Move the label and winner across row/chunk boundaries. The last row is a true tie, whose
    # step splits evenly between the tied competitors.
    z = np.full((4, classes), -100.0)
    labels = np.array([0, classes - 1, 1, 0], dtype=np.int64)
    expected = np.zeros_like(z)
    for row, label in enumerate(labels):
        competitors = [j for j in range(classes) if j != label]
        low, high = competitors[0], competitors[-1]
        z[row, label] = 0
        z[row, low] = 2.0**power
        z[row, high] = 2.0 ** (power + (row != 3))
        if row == 3:
            expected[row, low] = expected[row, high] = 0.5
        else:
            expected[row, high] = 1
        expected[row, label] = -1
    return z, labels, expected


def short_smooth_rows(classes: int, gamma: float, count: int):
    """Saturated tie, free mass, support boundary, masked, ignored and zero-weight rows."""
    scores = np.full((6, classes), -np.inf, dtype=np.float32)
    scores[:, 0] = 0
    scores[0, 1:3] = 2 * gamma - 1
    scores[1, 1:3] = (gamma / 2 - 1, gamma / 4 - 1)
    scores[2, 1:3] = (2 * gamma - 1, gamma - 1)
    scores[4, 0] = np.nan
    scores[5, -1] = 0
    labels = np.array([0, 0, 0, 0, -100, classes - 1], dtype=np.int32)
    weights = np.ones(classes, dtype=np.float32)
    weights[0], weights[-1] = 2, 0
    upstream = np.array([-0.375, 3, 0.5, -2, 1, 0], dtype=np.float32)
    values = gamma * np.array([3.5, 5 / 16, 3, 0, 0, 0], dtype=np.float32)
    gradient = np.zeros_like(scores)
    gradient[0, :3] = (0.75, -0.375, -0.375)
    gradient[1, :3] = (-4.5, 3, 1.5)
    gradient[2, :2] = (-1, 1)
    pattern = np.arange(count) % len(scores)
    return (
        scores[pattern],
        labels[pattern],
        weights,
        upstream[pattern],
        values[pattern],
        gradient[pattern],
    )


def wide_smooth_tiles(classes: int):
    """Separated winners, a split support, a crowded replay and an unprojected row."""
    z = np.full((6, classes), -8.0, np.float32)
    z[:, 0] = 0
    z[0, -1] = 3
    z[1, [1, -1]] = 3
    z[2, 1:17] = 2
    z[3, 1] = -0.875
    # The runner lies exactly gamma below the isolated maximum. The next
    # row has four positive masses and a fifth competitor at the exact cutoff.
    z[4, [1, -1]] = [3.0, 2.5]
    z[5, [1, 1025, 2049, 3073, -1]] = [10.8125, 10.8125, 10.6875, 10.6875, 10.625]
    seed = np.array([-1.75, 131072.0, -4.0, 0.25, -2.5, 1.75], np.float32)
    grad = np.zeros_like(z)
    grad[:, 0] = -seed
    grad[3, 0] = -seed[3] / 4
    grad[0, -1] = seed[0]
    grad[1, [1, -1]] = seed[1] / 2
    grad[2, 1:17] = seed[2] / 16
    grad[3, 1] = seed[3] / 4
    grad[4, 1] = seed[4]
    grad[5, [1, 1025]] = seed[5] * 0.375
    grad[5, [2049, 3073]] = seed[5] * 0.125
    values = np.array([3.75, 3.875, 3.0 - 0.25 / 16, 0.125**2, 3.75, 749 / 64], np.float32)
    return z, np.zeros(6, np.int64), seed, values, grad

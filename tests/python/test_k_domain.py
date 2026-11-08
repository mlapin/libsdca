"""Every loss checks 1 <= k < classes in the NumPy API and the scikit-learn estimator,
also the losses that do not use k (docs/API.md, parameter domains)."""

import importlib.util

import libsdca
import numpy as np
import pytest

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
CLASSES = 5
OUT_OF_RANGE = [0, CLASSES, 2 * CLASSES]
VALID = [1, CLASSES - 1]


def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 3))
    y = np.arange(20) % CLASSES
    return x, y, rng.normal(size=(20, CLASSES))


def targets(loss: str, y: np.ndarray) -> np.ndarray:
    return np.eye(CLASSES, dtype=bool)[y] if loss.startswith("multilabel") else y


def test_loss_and_grad(subtests: pytest.Subtests):
    _, y, z = problem()
    for loss in libsdca.LOSSES:
        for k in OUT_OF_RANGE:
            with subtests.test(loss=loss, k=k), pytest.raises(ValueError, match="k must be"):
                libsdca.loss_and_grad(z, targets(loss, y), loss, k=k)
        for k in VALID:
            with subtests.test(loss=loss, k=k):
                values, _ = libsdca.loss_and_grad(z, targets(loss, y), loss, k=k)
                assert np.isfinite(values).all()
    # multiclass_hinge does not use k.
    hinge = [libsdca.loss_and_grad(z, y, "multiclass_hinge", k=k)[0] for k in VALID]
    np.testing.assert_array_equal(hinge[0], hinge[1])


def test_coordinate_update(subtests: pytest.Subtests):
    _, y, z = problem()
    for objective in libsdca.OBJECTIVES:
        for k in OUT_OF_RANGE:
            with subtests.test(objective=objective, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    libsdca.coordinate_update(z, targets(objective, y), objective, k=k)
        with subtests.test(objective=objective, k=CLASSES - 1):
            out = libsdca.coordinate_update(z, targets(objective, y), objective, k=CLASSES - 1)
            assert np.isfinite(out).all()


def test_fit(subtests: pytest.Subtests):
    x, y, _ = problem()
    for objective in libsdca.OBJECTIVES:
        for k in OUT_OF_RANGE:
            with subtests.test(objective=objective, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    libsdca.fit(x, targets(objective, y), objective, k=k, max_epoch=2)
        with subtests.test(objective=objective, k=CLASSES - 1):
            libsdca.fit(x, targets(objective, y), objective, k=CLASSES - 1, max_epoch=2)


@pytest.mark.skipif(not HAVE_SKLEARN, reason="scikit-learn is not installed")
def test_sklearn_classifier(subtests: pytest.Subtests):
    from libsdca.sklearn import SDCAClassifier

    x, y, _ = problem()
    for objective in libsdca.OBJECTIVES:
        if objective.startswith("multilabel"):
            continue
        for k in OUT_OF_RANGE:
            with subtests.test(objective=objective, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    SDCAClassifier(objective, k=k, max_epoch=2).fit(x, y)
        with subtests.test(objective=objective, k=CLASSES - 1):
            SDCAClassifier(objective, k=CLASSES - 1).fit(x, y)

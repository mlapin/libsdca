"""The top-k losses check 1 <= k < classes in the NumPy API and the scikit-learn estimator,
and the other losses take no k (docs/API.md, parameter domains)."""

import functools
import importlib.util
import warnings

import libsdca
import named_losses
import numpy as np
import pytest

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
CLASSES = 5
OUT_OF_RANGE = [CLASSES, 2 * CLASSES]
VALID = [1, CLASSES - 1]
# The top-k kinds (both variants of topk_hinge).
TOPK = named_losses.TOPK


def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 3))
    y = np.arange(20) % CLASSES
    return x, y, rng.normal(size=(20, CLASSES))


def test_loss_and_grad(subtests: pytest.Subtests):
    _, y, z = problem()
    for name in TOPK:
        cls = functools.partial(named_losses.numpy_loss, name)
        with pytest.raises(ValueError, match="k must be"):
            cls(k=0)
        for k in OUT_OF_RANGE:
            with subtests.test(loss=name, k=k), pytest.raises(ValueError, match="k must be"):
                libsdca.loss_and_grad(z, y, cls(k=k))
        for k in VALID:
            with subtests.test(loss=name, k=k):
                values, _ = libsdca.loss_and_grad(z, y, cls(k=k))
                assert np.isfinite(values).all()
    for name in set(libsdca.LOSSES) - libsdca.TOPK:
        with subtests.test(loss=name), pytest.raises(TypeError):
            libsdca.LOSSES[name](k=2)


def test_coordinate_update(subtests: pytest.Subtests):
    _, y, z = problem()
    for name in TOPK:
        cls = functools.partial(named_losses.numpy_loss, name)
        for k in OUT_OF_RANGE:
            with subtests.test(objective=name, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    libsdca.coordinate_update(z, y, cls(k=k))
        with subtests.test(objective=name, k=CLASSES - 1):
            out = libsdca.coordinate_update(z, y, cls(k=CLASSES - 1))
            assert np.isfinite(out).all()


def test_fit(subtests: pytest.Subtests):
    x, y, _ = problem()
    for name in TOPK:
        cls = functools.partial(named_losses.numpy_loss, name)
        for k in OUT_OF_RANGE:
            with subtests.test(objective=name, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    libsdca.fit(x, y, cls(k=k), max_epoch=2)
        with subtests.test(objective=name, k=CLASSES - 1):
            libsdca.fit(x, y, cls(k=CLASSES - 1), max_epoch=2)


@pytest.mark.skipif(not HAVE_SKLEARN, reason="scikit-learn is not installed")
def test_sklearn_classifier(subtests: pytest.Subtests):
    from libsdca.sklearn import SDCAClassifier

    x, y, _ = problem()
    for name in sorted(libsdca.TOPK):
        for k in OUT_OF_RANGE:
            with subtests.test(objective=name, k=k):
                with pytest.raises(ValueError, match="k must be"):
                    SDCAClassifier(name, k=k, max_iter=2).fit(x, y)
        with subtests.test(objective=name, k=CLASSES - 1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SDCAClassifier(name, k=CLASSES - 1).fit(x, y)
    with pytest.raises(TypeError, match="takes no parameter k"):
        SDCAClassifier("multiclass_hinge", k=2).fit(x, y)

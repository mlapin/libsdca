"""Shrinking of the hinge objectives' epochs: the option, its certificate and its results."""

import ctypes as ct

import libsdca
import numpy as np
import pytest
from libsdca import HISTORY, STATUS
from libsdca._core import _default, _fit_result, _Options, _options, _ptr

MODES = ("auto", "off", "heuristic", "safe", "both")


def data(
    n: int = 300, d: int = 10, m: int = 6, seed: int = 7, dtype: type = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % m
    centers = rng.normal(size=(m, d)) * 1.5
    return (centers[y] + rng.normal(size=(n, d))).astype(dtype), y


def relative_gap(model: libsdca.Model) -> float:
    return float(model.history["relative_gap"][-1])


CASES: list[tuple[str, int, float]] = [
    ("multiclass_hinge", 1, 0.0),
    ("multiclass_hinge", 1, 0.5),
    ("topk_hinge_alpha", 2, 0.0),
    ("topk_hinge_beta", 2, 0.0),
    ("topk_hinge_beta", 2, 0.5),
]


@pytest.mark.parametrize("objective,k,gamma", CASES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_every_mode_certifies_the_same_optimum(objective: str, k: int, gamma: float, dtype: type):
    # precision="float64" keeps W and the duals of float32 data in double: the same tol.
    x, y = data(dtype=dtype)
    tol = 1e-6
    fits = {
        mode: libsdca.fit(
            x, y, objective, C=1.0, k=k, gamma=gamma, tol=tol, max_epoch=20000, shrinking=mode
        )
        for mode in MODES
    }
    for mode, model in fits.items():
        assert model.status == "solved", mode
        assert relative_gap(model) <= tol, mode
    reference = fits["off"].history["primal"][-1]
    for mode, model in fits.items():
        primal = model.history["primal"][-1]
        assert abs(primal - reference) <= 2 * tol * abs(reference), mode


def test_multilabel_and_kernel_fits_agree():
    x, y = data(n=120)
    labels = [[int(c), int((c + 1) % 6)] for c in y]
    for kernel in (False, True):
        fits = [
            libsdca.fit(
                x @ x.T if kernel else x,
                y if kernel else labels,
                "multiclass_hinge" if kernel else "multilabel_hinge",
                C=0.5,
                tol=1e-6,
                max_epoch=20000,
                kernel=kernel,
                shrinking=mode,
            )
            for mode in ("off", "both")
        ]
        assert all(f.status == "solved" for f in fits)
        p0, p1 = (f.history["primal"][-1] for f in fits)
        assert abs(p0 - p1) <= 2e-6 * abs(p0)


@pytest.mark.parametrize("mode", MODES)
def test_fits_are_deterministic(mode: str):
    x, y = data()
    first = libsdca.fit(x, y, "multiclass_hinge", tol=1e-5, max_epoch=5000, shrinking=mode)
    second = libsdca.fit(x, y, "multiclass_hinge", tol=1e-5, max_epoch=5000, shrinking=mode)
    assert np.array_equal(first.dual, second.dual)
    assert first.weights is not None and second.weights is not None
    assert np.array_equal(first.weights, second.weights)
    assert first.epochs == second.epochs


def test_a_fit_solved_at_max_epoch_reports_it():
    # The heuristic's evaluations are spaced by work; the one at max_epoch is not.
    x, y = data(n=1000, d=20, m=10)
    tol = 1e-6
    solved = libsdca.fit(x, y, "multiclass_hinge", tol=tol, max_epoch=20000)
    assert solved.status == "solved"
    for max_epoch in range(10, solved.epochs + 10, 10):
        model = libsdca.fit(x, y, "multiclass_hinge", tol=tol, max_epoch=max_epoch)
        assert model.epochs == min(max_epoch, solved.epochs), max_epoch
        if model.status == "max_epoch":
            assert relative_gap(model) > tol / 2, max_epoch


@pytest.mark.parametrize("objective", ["cross_entropy", "topk_cross_entropy"])
def test_entropy_objectives_do_not_shrink(objective: str):
    x, y = data()
    k = 2 if objective == "topk_cross_entropy" else 1
    fits = [
        libsdca.fit(x, y, objective, k=k, tol=1e-8, max_epoch=2000, shrinking=mode)
        for mode in MODES
    ]
    for model in fits[1:]:
        assert np.array_equal(model.dual, fits[0].dual)
        assert model.epochs == fits[0].epochs


def test_shrinking_saves_steps_on_hinges():
    # The examples beyond their margins: the default mode reaches the gap in a
    # fraction of the time per epoch; here, the same epochs give the same gap.
    x, y = data(n=2000, d=20, m=10)
    off = libsdca.fit(x, y, "multiclass_hinge", tol=1e-6, max_epoch=20000, shrinking="off")
    auto = libsdca.fit(x, y, "multiclass_hinge", tol=1e-6, max_epoch=20000)
    assert off.status == auto.status == "solved"
    assert auto.epochs <= 2 * off.epochs


def test_float32_digits_reach_float64_tolerances():
    # Float storage of W and the duals stalled at a gap of 4.5e-6 here (lost small updates);
    # precision="float64" keeps them in double and certifies the returned float32 arrays.
    datasets = pytest.importorskip("sklearn.datasets")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    x, y = datasets.load_digits(return_X_y=True)
    x = preprocessing.StandardScaler().fit_transform(x).astype(np.float32)
    model = libsdca.fit(x, y, "multiclass_hinge", C=1.0, tol=1e-6, max_epoch=4000)
    assert model.status == "solved" and relative_gap(model) <= 1e-6
    assert model.weights is not None and model.weights.dtype == model.dual.dtype == np.float32
    w = model.weights.astype(np.float64)
    z = x.astype(np.float64) @ w
    rows = np.arange(len(y))
    h = 1 + z - z[rows, y][:, None]
    h[rows, y] = 0
    primal = 0.5 * float((w * w).sum()) + float(h.max(1).sum())
    assert abs(primal - model.history["primal"][-1]) <= 1e-12 * primal


@pytest.mark.parametrize("value", ["on", "true", "", None, 1, True])
def test_invalid_values_are_rejected(value: object):
    x, y = data(n=20)
    with pytest.raises(ValueError, match="shrinking must be one of"):
        libsdca.fit(x, y, "multiclass_hinge", shrinking=value)  # pyright: ignore[reportArgumentType]


def test_sklearn_takes_the_option():
    from libsdca.sklearn import SDCAClassifier

    x, y = data(n=200)
    clf = SDCAClassifier("multiclass_hinge", tol=1e-6, max_epoch=20000, shrinking="safe")
    assert clf.get_params()["shrinking"] == "safe"
    clf.fit(x, y)
    off = SDCAClassifier("multiclass_hinge", tol=1e-6, max_epoch=20000, shrinking="off").fit(x, y)
    assert clf.status_ == off.status_ == "solved"
    p0, p1 = clf.history_["primal"][-1], off.history_["primal"][-1]
    assert abs(p0 - p1) <= 2e-6 * abs(p1)
    with pytest.raises(ValueError, match="shrinking"):
        SDCAClassifier("multiclass_hinge", shrinking="maybe").fit(x, y)


def test_the_struct_of_an_earlier_header_takes_auto():
    # sdca_options of the headers before shrinking: the size up to the field. The epoch
    # budget leaves room for the platform's path (std::shuffle's order is the library's).
    solver = _default("float64")
    x, y = data(n=40, d=3, m=3)
    w, a = np.zeros((3, 3)), np.zeros((40, 3))
    hist = np.zeros((10002, len(HISTORY)))
    labels = y.astype(np.int64)
    args = (40, 3, 3, _ptr(x), _ptr(labels), None, None, _ptr(w), _ptr(a), _ptr(hist))
    weights = []
    cases = ((_Options.shrinking.offset, 99), (ct.sizeof(_Options), 0), (ct.sizeof(_Options), 1))
    for size, shrinking in cases:
        options = _options(loss=2, max_epoch=10000, tol=1e-6, shrinking=shrinking)
        options.size = size
        result = _fit_result()
        w[:] = 0
        a[:] = 0
        solver._call("sdca_fit", ct.byref(options), *args, len(hist), ct.byref(result))
        assert result.status == STATUS.index("solved")
        weights.append(w.copy())
    np.testing.assert_array_equal(weights[0], weights[1])
    options = _options(loss=2, shrinking=5)
    with pytest.raises(ValueError, match="Invalid shrinking"):
        solver._call("sdca_fit", ct.byref(options), *args, len(hist), ct.byref(_fit_result()))

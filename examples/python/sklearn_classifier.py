"""SDCAClassifier as a scikit-learn estimator: fit/predict, Pipeline, GridSearchCV, kernels."""

import numpy as np
from libsdca import SDCAClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 20)) * rng.uniform(0.1, 10, size=20)  # unscaled features
labels = (X / X.std(axis=0) @ rng.normal(size=(20, 4))).argmax(axis=1)
y = np.array(["ant", "bee", "cat", "dog"])[labels]  # any sortable labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SDCAClassifier("cross_entropy", C=1.0).fit(StandardScaler().fit_transform(X_train), y_train)
assert clf.coef_ is not None  # None only for precomputed kernels
print(clf.classes_, clf.coef_.shape, clf.status_, clf.n_iter_)

# A Pipeline with scaling, and a grid over objective, k and C.
pipe = make_pipeline(StandardScaler(), SDCAClassifier(tol=1e-4))
grid = GridSearchCV(
    pipe,
    {
        "sdcaclassifier__objective": ["cross_entropy", "topk_cross_entropy", "topk_hinge_beta"],
        "sdcaclassifier__k": [1, 2],
        "sdcaclassifier__C": [0.1, 1.0, 10.0],
    },
    cv=3,
)
grid.fit(X_train, y_train)
print("best", grid.best_params_, f"cv {grid.best_score_:.3f}")
test_accuracy = accuracy_score(y_test, grid.predict(X_test))
print(f"test accuracy {test_accuracy:.3f}")
assert test_accuracy > 0.8

# predict_proba exists for objective="cross_entropy".
proba = make_pipeline(StandardScaler(), SDCAClassifier("cross_entropy")).fit(X_train, y_train)
print(proba.predict_proba(X_test[:2]).round(3))

# A precomputed kernel: fit on K(train, train), predict from K(test, train).
scaler = StandardScaler().fit(X_train)
S_train, S_test = scaler.transform(X_train), scaler.transform(X_test)
K_train, K_test = rbf_kernel(S_train, gamma=0.02), rbf_kernel(S_test, S_train, gamma=0.02)
kernel = SDCAClassifier("topk_hinge_alpha", C=10.0, kernel="precomputed").fit(K_train, y_train)
print(f"rbf kernel test accuracy {kernel.score(K_test, y_test):.3f}")

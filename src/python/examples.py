###########################################################################
## Example 1

import numpy as np
import pandas as pd

from libsdca import libsdca
SEED = 1234
prng = np.random.RandomState(SEED)

# build datasets
k = 5
num_classes = 11
num_features = 100
num_observations = 200000

datasets = []
for i in range(1):
    f = (prng.rand(
        num_features, num_observations
    ) * 10. - 5.).astype(np.double) # must be type double
    l = prng.randint(
        1,num_classes, num_observations
    ).astype(np.int32) # must be type int32
    datasets.append((f,l))

# fit model
model = libsdca.py_solve(datasets, num_classes, log_level="verbose",
                         k=k, C=0.001, return_records=0, return_evals=0)

# report prediction scores
prediction_scores = pd.DataFrame(model.W.T.dot(datasets[1][0])).T

# select top-k predictions given scores
top_k = pd.DataFrame(prediction_scores.values.argsort()[:,-k:][:,::-1])

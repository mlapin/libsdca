# cython: wraparound = False
# cython: boundscheck = False

from __future__ import print_function, division

from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from py_prox cimport py_prox_, py_prox_inplace_, proxOpts
from py_solve cimport py_solve_, dataset, solveOpts, modelInfo

ctypedef dataset[double] DATASET

###########################################################################
## numpy goodness

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef np.ndarray[np.double_t, ndim=1, mode="c"] \
    carray_to_numpy_1d(np.npy_intp dims, double* ptr):
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] arr \
        = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef np.ndarray[np.double_t, ndim=2, mode="c"] \
    carray_to_numpy_2d(np.npy_intp* dims, double* ptr):
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] arr \
        = np.PyArray_SimpleNewFromData(2, &dims[0], np.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef np.ndarray[np.double_t, ndim=3, mode="c"] \
    carray_to_numpy_3d(np.npy_intp* dims, double* ptr):
    cdef np.ndarray[np.double_t, ndim=3, mode="c"] arr \
        = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

###########################################################################
## return class for py_solve

class ModelInfo(object):
    def __init__(self, A, W, primal, dual,
                 absolute_gap, relative_gap, epoch,
                 wall_time, cpu_time,
                 solve_wall_time, solve_cpu_time,
                 eval_wall_time, eval_cpu_time,
                 records=None, evals=None):

        self.A = A
        self.W = W
        self.records = records
        self.evals = evals
        self.primal = primal
        self.dual = dual
        self.absolute_gap = absolute_gap
        self.relative_gap = relative_gap
        self.epoch = epoch
        self.wall_time = wall_time
        self.cpu_time = cpu_time
        self.solve_wall_time = solve_wall_time
        self.solve_cpu_time = solve_cpu_time
        self.eval_wall_time = eval_wall_time
        self.eval_cpu_time = eval_cpu_time

###########################################################################
## interfaces to cpp code

cdef DATASET* build_dataset(double [:,:] features,
                            int [:] labels,
                            size_t num_classes):

    cdef DATASET* dat = new DATASET()
    dat.num_dimensions = features.shape[0]
    dat.num_examples = features.shape[1]
    dat.num_classes = num_classes
    dat.labels.assign(&labels[0], &labels[0] + features.shape[1])
    dat.data = &features[0,0]
    return &dat[0]

def py_solve(datasets, num_classes,
             int k=1, double C=1,
             str log_level="info", str log_format="short_f",
             str precision="double", str summation="default",
             str objective="topk_svm",
             int check_on_start=0, int is_dual=0,
             int check_epoch=10, int max_epoch=1000,
             double max_cpu_time=0, double max_wall_time=0.,
             double epsilon=1e-3, int return_records=0,
             int return_evals=0):
    # indent
    cdef size_t d = datasets[0][0].shape[0]
    cdef size_t n = datasets[0][0].shape[1]
    cdef size_t m = num_classes

    # initialize model info
    cdef modelInfo[double]* info = new modelInfo[double]()
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] W \
        = np.empty([d, m], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] A \
        = np.empty([m, n], dtype=np.double)
    info.W = &W[0,0]
    info.A = &A[0,0]

    # build vector of data sets
    cdef vector[DATASET] _datasets
    cdef DATASET* dat
    for f, l in datasets:
        dat = build_dataset(f, l, num_classes)
        _datasets.push_back(dat[0])

    # build solver options structure
    cdef solveOpts[double]* opts = new solveOpts[double](
        k = k,
        C = C,
        is_dual = is_dual,
        log_level = bytes(log_level, encoding="UTF-8"),
        log_format = bytes(log_format, encoding="UTF-8"),
        precision = bytes(precision, encoding="UTF-8"),
        summation = bytes(summation, encoding="UTF-8"),
        objective = bytes(objective, encoding="UTF-8"),
        check_on_start = check_on_start,
        check_epoch = check_epoch,
        max_epoch = max_epoch,
        max_cpu_time = max_cpu_time,
        max_wall_time = max_wall_time,
        epsilon = epsilon,
        return_records = return_records,
        return_evals = return_evals)

    with nogil:
        py_solve_[double](&_datasets, &info[0], &opts[0])

    model = ModelInfo(A, W, info.primal, info.dual,
                 info.absolute_gap, info.relative_gap, info.epoch,
                 info.wall_time, info.cpu_time,
                 info.solve_wall_time, info.solve_cpu_time,
                 info.eval_wall_time, info.eval_cpu_time)

    records = None
    cdef np.npy_intp* r_dims = [info.num_records, 13]
    if return_records:
        model.records = carray_to_numpy_2d(
            &r_dims[0], &info.records[0]
        )

    evals = None
    cdef np.npy_intp* e_dims = [
        info.num_dataset_evals, info.num_evals, 1 + num_classes
    ]
    if return_evals:
        model.evals = carray_to_numpy_3d(
            &e_dims[0], &info.evals[0])

    return model


cpdef py_prox(double[:,:] A, double[:,:] X, size_t k = 1,
          str summation="kahan", str prox="knapsack",
          int in_place=1,
          double hi=1, double lo=0,
          double rhs=1, double rho=1):
    # indent
    cdef size_t m = A.shape[0], n = A.shape[1]
    cdef proxOpts[double]* opts = new proxOpts[double](
        m = m,
        n = n,
        k = k,
        summation = bytes(summation, encoding="UTF-8"),
        prox = bytes(prox, encoding="UTF-8"),
        lo = lo,
        hi = hi,
        rhs = rhs,
        rho = rho)

    if in_place == 1:
        with nogil:
            # solution is written over in values of A
            py_prox_inplace_(&A[0,0], &opts[0])
    else:
        with nogil:
            # solution is written over in values of X
            py_prox_(&A[0,0], &X[0,0], &opts[0])

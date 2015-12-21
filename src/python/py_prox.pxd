from libcpp.string cimport string

cdef extern from "py_prox.h" namespace "sdca" nogil:
    cdef cppclass proxOpts[Result]:
        size_t m
        size_t n
        size_t k
        string summation
        string prox
        Result lo
        Result hi
        Result rhs
        Result rho
        proxOpts(size_t m, size_t n, size_t k,
                 string summation, string prox,
                 Result lo, Result hi,
                 Result rhs, Result rho) except +

cdef extern from "py_prox.h" namespace "py_prox" nogil:
    void py_prox_(
        double* A, double* X, proxOpts[double]* opts) except +
    void py_prox_inplace_(
        double* A, proxOpts[double]* opts) except +

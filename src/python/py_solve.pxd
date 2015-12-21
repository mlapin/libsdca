from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "py_solve.h" namespace "sdca" nogil:
    cdef cppclass dataset[Data]:
        size_t num_dimensions
        size_t num_examples
        size_t num_classes
        vector[size_t] labels
        Data* data
        string to_string() const

    cdef cppclass solveOpts[Result]:
        size_t k
        Result C
        int is_dual
        string log_level
        string log_format
        string precision
        string summation
        string objective
        int check_on_start
        size_t check_epoch
        size_t max_epoch
        Result max_cpu_time
        Result max_wall_time
        Result epsilon
        int return_records
        int return_evals
        solveOpts(size_t k, Result C, int is_dual,
                  string log_level, string log_format,
                  string precision, string summation,
                  string objective, int check_on_start,
                  int check_epoch, size_t max_epoch,
                  Result max_cpu_time, Result max_wall_time,
                  Result epsilon, int return_records,
                  int return_evals)

    cdef cppclass modelInfo[Result]:
        Result* A
        Result* W
        Result* records # num_records x 13
        size_t num_records
        Result* records # num_records x 13
        size_t num_dataset_evals
        size_t num_evals
        Result* evals # num_evals x num_dataset_evals x (1 + k)
        Result primal
        Result dual
        Result absolute_gap
        Result relative_gap
        size_t epoch
        Result wall_time
        Result cpu_time
        Result solve_wall_time
        Result solve_cpu_time
        Result eval_wall_time
        Result eval_cpu_time

cdef extern from "py_solve.h" namespace "py_solve" nogil:
    void py_solve_[Result](
        vector[dataset[double]]* datasets, modelInfo[Result]* info, solveOpts[Result]* opts)

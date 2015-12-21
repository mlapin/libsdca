#ifndef SDCA_PYTHON_PY_UTIL_H
#define SDCA_PYTHON_PY_UTIL_H

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "solve/solvedef.h"
#include "util/logging.h"

namespace sdca {

// indent

template <typename Result>
struct proxOpts {
  size_t m;
  size_t n;
  size_t k;
  std::string summation;
  std::string prox;
  Result lo;
  Result hi;
  Result rhs;
  Result rho;
  proxOpts(size_t m__,
           size_t n__,
           size_t k__ = 1,
           std::string summation__ = "standard",
           std::string prox__ = "knapsack",
           Result lo__ = 0,
           Result hi__ = 1,
           Result rhs__ = 0,
           Result rho__ = 0
      ) :
        m(m__),
        n(n__),
        k(k__),
        summation(summation__),
        prox(prox__),
        lo(lo__),
        hi(hi__),
        rhs(rhs__),
        rho(rho__)
    {}
};

template <typename Result>
struct solveOpts {
  size_t k;
  Result C;
  bool is_dual;
  std::string log_level;
  std::string log_format;
  std::string precision;
  std::string summation;
  std::string objective;
  bool check_on_start;
  size_t check_epoch;
  size_t max_epoch;
  Result max_cpu_time;
  Result max_wall_time;
  Result epsilon;
  bool return_records;
  bool return_evals;
  solveOpts(size_t k__ = 1,
            Result C__ = 1,
            bool is_dual__ = false,
            std::string log_level__ = "info",
            std::string log_format__ = "short_f",
            std::string precision__ = "double",
            std::string summation__ = "default",
            std::string objective__ = "topk_svm",
            bool check_on_start__ = false,
            size_t check_epoch__ = 10,
            size_t max_epoch__ = 1000,
            Result max_cpu_time__ = 0,
            Result max_wall_time__ = 0,
            Result epsilon__ = 1e-3,
            bool return_records__ = false,
            bool return_evals__ = false
      ) :
        k(k__),
        C(C__),
        is_dual(is_dual__),
        log_level(log_level__),
        log_format(log_format__),
        precision(precision__),
        summation(summation__),
        objective(objective__),
        check_on_start(check_on_start__),
        check_epoch(check_epoch__),
        max_epoch(max_epoch__),
        max_cpu_time(max_cpu_time__),
        max_wall_time(max_wall_time__),
        epsilon(epsilon__),
        return_records(return_records__),
        return_evals(return_evals__)
    {}
};

template <typename Result>
struct modelInfo {
  size_t num_examples;
  size_t num_dimensions;
  size_t num_classes;
  size_t k;
  Result* A;
  Result* W;
  size_t num_records;
  Result* records; // num_records x 13
  size_t num_dataset_evals;
  size_t num_evals;
  Result* evals; // evals[0].size() x evals.size() x (1 + k)
  Result C;
  bool is_dual;
  std::string log_level;
  std::string log_format;
  std::string precision;
  std::string summation;
  std::string objective;
  bool check_on_start;
  size_t check_epoch;
  size_t max_epoch;
  Result max_cpu_time;
  Result max_wall_time;
  Result epsilon;
  std::string status;
  Result primal;
  Result dual;
  Result absolute_gap;
  Result relative_gap;
  size_t epoch;
  Result wall_time;
  Result cpu_time;
  Result solve_wall_time;
  Result solve_cpu_time;
  Result eval_wall_time;
  Result eval_cpu_time;
};

template <typename Result>
inline void
set_logging_options(
    solveOpts<Result>* opts
  ) {
  if (opts->log_level == "none") {
    logging::set_level(logging::none);
  } else if (opts->log_level == "info") {
    logging::set_level(logging::info);
  } else if (opts->log_level == "verbose") {
    logging::set_level(logging::verbose);
  } else if (opts->log_level == "debug") {
    logging::set_level(logging::debug);
  } else {
    throw std::runtime_error(
        "invalid value passed to logging_options"
        " in opts.log_level.  Valid options are: "
        "none, info, verbose, or debug."
        );
  }
  if (opts->log_format == "short_f") {
    logging::set_format(logging::short_f);
  } else if (opts->log_format == "short_e") {
    logging::set_format(logging::short_e);
  } else if (opts->log_format == "long_f") {
    logging::set_format(logging::long_f);
  } else if (opts->log_format == "long_e") {
    logging::set_format(logging::long_e);
  } else {
    throw std::runtime_error(
        "Invalid value passed to logging_options"
        " in opts.log_format.  Valid options are: "
        "short_F, short_e, long_f, or long_e."
        );
  }
}

template <typename Data>
inline void
set_datasets(
    std::vector<dataset<Data>>* datasets,
    solver_context<Data>& context
  ) {
  // Testing datasets
  size_t num_datasets = datasets->size();
  for (size_t i = 0; i < num_datasets; ++i) {
      context.datasets.push_back((*datasets)[i]);
  }
}

template <typename Data,
          typename Result>
void
set_stopping_criteria(
    solveOpts<Result>* opts,
    solver_context<Data>& context
    ) {
  auto c = &context.criteria;
  c->check_on_start = opts->check_on_start;
  c->check_epoch = opts->check_epoch;
  c->max_epoch = opts->max_epoch;
  c->max_cpu_time = opts->max_cpu_time;
  c->max_wall_time = opts->max_wall_time;
  c->epsilon = opts->epsilon;
}

}

#endif

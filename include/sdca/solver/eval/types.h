#ifndef SDCA_SOLVER_EVAL_TYPES_H
#define SDCA_SOLVER_EVAL_TYPES_H

#include <iterator>
#include <limits>
#include <iostream>

#include "sdca/solver/data/output.h"

namespace sdca {

template <typename Result>
struct eval_train_base {

  Result primal = Result();
  Result dual = Result();
  Result primal_loss = Result();
  Result dual_loss = Result();
  Result regularizer = Result();

  double solve_time_cpu = 0;
  double solve_time_wall = 0;
  double eval_time_cpu = 0;
  double eval_time_wall = 0;
  size_type epoch = 0;


  inline Result absolute_gap() const {
    return primal - dual;
  }


  inline Result relative_gap() const {
    Result max = std::max(std::abs(primal), std::abs(dual));
    return (max > static_cast<Result>(0))
      ? (max < std::numeric_limits<Result>::infinity()
        ? (primal - dual) / max
        : std::numeric_limits<Result>::infinity())
      : static_cast<Result>(0);
  }


  double cpu_time() const {
    return solve_time_cpu + eval_time_cpu;
  }


  double wall_time() const {
    return solve_time_wall + eval_time_wall;
  }


  inline std::string to_string(bool skip_time = false) const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "epoch: " << epoch << ", "
    "relative_gap: " << relative_gap() << ", "
    "absolute_gap: " << absolute_gap() << ", "
    "primal: " << primal << ", "
    "dual: " << dual << ", "
    "primal_loss: " << primal_loss << ", "
    "dual_loss: " << dual_loss << ", "
    "regularizer: " << regularizer;
    if (!skip_time) {
      str << ", cpu_time: " << cpu_time() <<
               " (solve: " << solve_time_cpu <<
               ", eval: " << eval_time_cpu << ")"
             ", wall_time: " << wall_time() <<
               " (solve: " << solve_time_wall <<
               ", eval: " << eval_time_wall << ")";
    }
    return str.str();
  }
};


template <typename Result,
          typename Output>
struct eval_train {};


template <typename Result>
struct eval_train<Result, multiclass_output>
    : public eval_train_base<Result> {

  typedef eval_train_base<Result> base;

  std::vector<Result> accuracy;


  inline Result topk_accuracy(size_type k) const {
    return (k < accuracy.size()) ? accuracy[k] : 1;
  }


  inline std::string to_string(bool skip_time = false) const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << ", accuracy: ";
    long offset = std::min(5L, static_cast<long>(accuracy.size()));
    std::copy(accuracy.begin(), accuracy.begin() + offset,
      std::ostream_iterator<Result>(str, " "));
    return base::to_string(skip_time) + str.str();
  }
};


template <typename Result>
struct eval_train<Result, multilabel_output>
    : public eval_train_base<Result> {

  typedef eval_train_base<Result> base;

  Result rank_loss = Result();


  inline std::string to_string(bool skip_time = false) const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << ", rank_loss: " << rank_loss;
    return base::to_string(skip_time) + str.str();
  }
};


template <typename Result,
          typename Output>
struct eval_test {};


template <typename Result>
struct eval_test<Result, multiclass_output> {

  Result primal_loss = Result();
  std::vector<Result> accuracy;


  inline Result topk_accuracy(size_type k) const {
    return (k < accuracy.size()) ? accuracy[k] : 1;
  }


  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "primal_loss: " << primal_loss << ", "
           "accuracy: ";
    long offset = std::min(5L, static_cast<long>(accuracy.size()));
    std::copy(accuracy.begin(), accuracy.begin() + offset,
      std::ostream_iterator<Result>(str, " "));
    return str.str();
  }
};


template <typename Result>
struct eval_test<Result, multilabel_output> {

  Result primal_loss = Result();
  Result rank_loss = Result();


  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "primal_loss: " << primal_loss << ", "
           "rank_loss: " << rank_loss;
    return str.str();
  }
};

}

#endif

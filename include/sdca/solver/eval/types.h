#ifndef SDCA_SOLVER_EVAL_TYPES_H
#define SDCA_SOLVER_EVAL_TYPES_H

#include <iostream>
#include <limits>

#include "sdca/solver/output.h"

namespace sdca {

template <typename Result,
          typename Output>
struct eval_train {};


template <typename Result>
struct eval_train<Result, multiclass_output> {

  Result primal = Result();
  Result dual = Result();
  Result primal_loss = Result();
  Result dual_loss = Result();
  Result regularizer = Result();
  std::vector<Result> accuracy;


  inline Result relative_gap() const {
    Result max = std::max(std::abs(primal), std::abs(dual));
    return (max > static_cast<Result>(0))
      ? (max < std::numeric_limits<Result>::infinity()
        ? (primal - dual) / max
        : std::numeric_limits<Result>::infinity())
      : static_cast<Result>(0);
  }


  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "accuracy: ";
    long offset = std::min(5L, static_cast<long>(accuracy.size()));
    std::copy(accuracy.begin(), accuracy.begin() + offset,
      std::ostream_iterator<Result>(str, ", "));
    str << "relative_gap: " << relative_gap() << ", "
    "absolute_gap: " << primal - dual << ", "
    "primal: " << primal << ", "
    "dual: " << dual << ", "
    "primal_loss: " << primal_loss << ", "
    "dual_loss: " << dual_loss << ", "
    "regularizer: " << regularizer;
    return str.str();
  }
};


template <typename Result,
          typename Output>
struct eval_test {};


template <typename Result>
struct eval_test<Result, multiclass_output> {

  Result primal_loss = Result();
  std::vector<Result> accuracy;


  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "accuracy: ";
    long offset = std::min(5L, static_cast<long>(accuracy.size()));
    std::copy(accuracy.begin(), accuracy.begin() + offset,
      std::ostream_iterator<Result>(str, ", "));
    str << "primal_loss: " << primal_loss;
    return str.str();
  }
};

}

#endif

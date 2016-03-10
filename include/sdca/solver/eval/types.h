#ifndef SDCA_SOLVER_EVAL_TYPES_H
#define SDCA_SOLVER_EVAL_TYPES_H

#include <limits>

#include "sdca/solver/output.h"

namespace sdca {

template <typename Result,
          typename Output>
struct eval_train {
  typedef Result result_type;
  typedef Output output_type;

  Result primal = Result();
  Result dual = Result();
  Result primal_loss = Result();
  Result dual_loss = Result();
  Result regularizer = Result();


  inline Result relative_gap() const {
    Result max = std::max(std::abs(primal), std::abs(dual));
    return (max > static_cast<Result>(0))
      ? (max < std::numeric_limits<Result>::infinity()
        ? (primal - dual) / max
        : std::numeric_limits<Result>::infinity())
      : static_cast<Result>(0);
  }
};


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
};


template <typename Result,
          typename Output>
struct eval_test {
  typedef Result result_type;
  typedef Output output_type;

  Result primal_loss = Result();

};


template <typename Result>
struct eval_test<Result, multiclass_output> {

  Result primal_loss = Result();
  std::vector<Result> accuracy;

};

}

#endif

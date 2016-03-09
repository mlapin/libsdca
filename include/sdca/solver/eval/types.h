#ifndef SDCA_SOLVER_EVAL_TYPES_H
#define SDCA_SOLVER_EVAL_TYPES_H

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

};


template <typename Result>
struct eval_train<Result, multiclass_output> {

  Result primal = Result();
  Result dual = Result();
  Result primal_loss = Result();
  Result dual_loss = Result();
  Result regularizer = Result();
  std::vector<Result> accuracy;

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

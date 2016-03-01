#ifndef SDCA_SOLVER_EVAL_TYPE_H
#define SDCA_SOLVER_EVAL_TYPE_H

#include "sdca/solver/output.h"

namespace sdca {

template <typename Result,
          typename Output = multiclass_output>
struct eval_train {
  typedef Result result_type;
  typedef Output output_type;

  Result primal;
  Result dual;
  Result primal_loss;
  Result dual_loss;
  Result regularizer;

};


template <typename Result>
struct eval_train<Result, multiclass_output> {

  Result primal;
  Result dual;
  Result primal_loss;
  Result dual_loss;
  Result regularizer;
  std::vector<Result> accuracy;

};


template <typename Result,
          typename Output = multiclass_output>
struct eval_test {
  typedef Result result_type;
  typedef Output output_type;

  Result loss;

};


template <typename Result>
struct eval_test<Result, multiclass_output> {

  Result loss;
  std::vector<Result> accuracy;

};

}

#endif

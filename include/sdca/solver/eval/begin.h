#ifndef SDCA_SOLVER_EVAL_BEGIN_H
#define SDCA_SOLVER_EVAL_BEGIN_H

#include "sdca/solver/dataset.h"

namespace sdca {

template <typename Result,
          typename Input,
          template <typename, typename> class Evaluation>
inline Evaluation<Result, multiclass_output>&
eval_begin(
    eval_dataset<Input, multiclass_output,
                 Evaluation<Result, multiclass_output>>& d
  ) {
  // Create a new eval using the default c'tor
  d.evals.resize(d.evals.size() + 1);
  Evaluation<Result, multiclass_output>& eval = d.evals.back();

  // Allocate storage for the multiclass performance metric
  eval.accuracy.resize(d.out.num_classes);
  return eval;
}

}

#endif

#ifndef SDCA_SOLVER_OBJECTIVE_H
#define SDCA_SOLVER_OBJECTIVE_H

#include <cstddef>

#include "sdca/solver/objective/objective_base.h"
#include "sdca/solver/objective/l2_entropy_topk.h"
#include "sdca/solver/objective/l2_hinge_topk.h"
#include "sdca/solver/objective/l2_topk_hinge.h"

namespace sdca {


template <typename Data,
          typename Result = double>
inline l2_entropy_topk<Data, Result>
make_objective_l2_entropy_topk(
    const Result c = 1,
    const size_type k = 1
  ) {
  return l2_entropy_topk<Data, Result>(c, k);
}


template <typename Data,
          typename Result = double>
inline l2_hinge_topk<Data, Result>
make_objective_l2_hinge_topk(
    const Result c = 1,
    const size_type k = 1
  ) {
  return l2_hinge_topk<Data, Result>(c, k);
}


template <typename Data,
          typename Result = double>
inline l2_hinge_topk_smooth<Data, Result>
make_objective_l2_hinge_topk_smooth(
    const Result c = 1,
    const Result gamma = 1,
    const size_type k = 1
  ) {
  return l2_hinge_topk_smooth<Data, Result>(c, gamma, k);
}


template <typename Data,
          typename Result = double>
inline l2_topk_hinge<Data, Result>
make_objective_l2_topk_hinge(
    const Result c = 1,
    const size_type k = 1
  ) {
  return l2_topk_hinge<Data, Result>(c, k);
}


template <typename Data,
          typename Result = double>
inline l2_topk_hinge_smooth<Data, Result>
make_objective_l2_topk_hinge_smooth(
    const Result c = 1,
    const Result gamma = 1,
    const size_type k = 1
  ) {
  return l2_topk_hinge_smooth<Data, Result>(c, gamma, k);
}


}

#endif

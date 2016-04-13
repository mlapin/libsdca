#ifndef SDCA_SOLVER_OBJECTIVE_H
#define SDCA_SOLVER_OBJECTIVE_H

#include "sdca/solver/objective/l2_entropy_topk.h"
#include "sdca/solver/objective/l2_hinge_topk.h"
#include "sdca/solver/objective/l2_multilabel_entropy.h"
#include "sdca/solver/objective/l2_multilabel_hinge.h"
#include "sdca/solver/objective/l2_topk_hinge.h"

namespace sdca {

template <typename Objective>
struct has_param_k : std::false_type {};

template <typename Data,
          typename Result>
struct has_param_k<l2_entropy_topk<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_k<l2_hinge_topk<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_k<l2_hinge_topk_smooth<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_k<l2_topk_hinge<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_k<l2_topk_hinge_smooth<Data, Result>> : std::true_type {};


template <typename Objective>
struct has_param_gamma : std::false_type {};

template <typename Data,
          typename Result>
struct has_param_gamma<l2_hinge_topk_smooth<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_gamma<l2_topk_hinge_smooth<Data, Result>> : std::true_type {};

template <typename Data,
          typename Result>
struct has_param_gamma<l2_multilabel_hinge_smooth<Data, Result>>
  : std::true_type {};


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


template <typename Data,
          typename Result = double>
inline l2_multilabel_entropy<Data, Result>
make_objective_l2_multilabel_entropy(
    const Result c = 1
  ) {
  return l2_multilabel_entropy<Data, Result>(c);
}


template <typename Data,
          typename Result = double>
inline l2_multilabel_hinge<Data, Result>
make_objective_l2_multilabel_hinge(
    const Result c = 1
  ) {
  return l2_multilabel_hinge<Data, Result>(c);
}


template <typename Data,
          typename Result = double>
inline l2_multilabel_hinge_smooth<Data, Result>
make_objective_l2_multilabel_hinge_smooth(
    const Result c = 1,
    const Result gamma = 1
  ) {
  return l2_multilabel_hinge_smooth<Data, Result>(c, gamma);
}

}

#endif

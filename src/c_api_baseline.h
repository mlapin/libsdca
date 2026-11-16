#ifndef SDCA_SRC_C_API_BASELINE_H
#define SDCA_SRC_C_API_BASELINE_H

// SDCA_BASELINE: the training entry points of src/c_api.cpp over earlier headers
// (the 2016 headers fd5c1f1 and the 0.3 trees), for benchmarks against earlier
// releases only. It gives those headers the names src/c_api.cpp uses:
// the objective factories of the shared loss names, the feature context of
// (W, A, center), and the stopping criteria of the fit options. Their solvers
// have no seed, threads, shrinking, projection precision or primal rebuild:
// those options keep their defaults there.

#include <cstddef>
#include <stdexcept>

#include "sdca/solver.h"
#include "sdca/utility/logging.cpp"  // the definitions of the earlier logging globals

namespace sdca {

template <typename D, typename R>
l2_entropy<D, R> make_objective_l2_cross_entropy(R c) {
  return l2_entropy<D, R>(c);
}
template <typename D, typename R>
l2_entropy_topk<D, R> make_objective_l2_topk_cross_entropy(R c, size_type k) {
  return l2_entropy_topk<D, R>(c, k);
}
template <typename D, typename R>
l2_hinge_topk<D, R> make_objective_l2_topk_hinge_alpha(R c, size_type k) {
  return l2_hinge_topk<D, R>(c, k);
}
// The earlier smooth hinges take gamma before k.
template <typename D, typename R>
l2_hinge_topk_smooth<D, R> make_objective_l2_topk_hinge_alpha_smooth(R c, size_type k, R gamma) {
  return l2_hinge_topk_smooth<D, R>(c, gamma, k);
}
template <typename D, typename R>
l2_topk_hinge<D, R> make_objective_l2_topk_hinge_beta(R c, size_type k) {
  return l2_topk_hinge<D, R>(c, k);
}
template <typename D, typename R>
l2_topk_hinge_smooth<D, R> make_objective_l2_topk_hinge_beta_smooth(R c, size_type k, R gamma) {
  return l2_topk_hinge_smooth<D, R>(c, gamma, k);
}
// make_objective_l2_multilabel_hinge[_smooth] keep their names.
template <typename D, typename R>
l2_multilabel_entropy<D, R> make_objective_l2_multilabel_cross_entropy(R c) {
  return l2_multilabel_entropy<D, R>(c);
}

// The feature context of the current headers: W before A, and the center.
template <typename Data, typename Result, typename Output,
          template <typename, typename> class Objective>
solver_context<Data, Result, feature_input, Output, Objective> make_context(
    feature_input<Data>&& in, Output&& out, Objective<Data, Result>&& objective,
    Data* primal_variables, Data* dual_variables, const Data* center) {
  auto ctx = make_context(std::move(in), std::move(out), std::move(objective), dual_variables,
                          primal_variables);
  ctx.primal_initial = center;
  return ctx;
}

}  // namespace sdca

namespace {

const bool quiet_logger = [] {
  sdca::logging::set_level(sdca::logging::level::none);
  return true;
}();

void configure(sdca::stopping_criteria& criteria, const sdca_fit_options& o, const sdca_loss&,
               std::size_t) {
  if (o.seed != 0 || o.recompute_primal != 1 || o.shrinking != SDCA_SHRINKING_AUTO)
    throw std::invalid_argument("The earlier headers take no seed, shrinking or primal rebuild");
  criteria.max_epoch = o.max_epoch;
  criteria.eval_epoch = o.eval_epoch;
  criteria.epsilon = o.tol;
  criteria.eval_on_start = o.eval_on_start != 0;
}

template <typename D, typename Objective, typename Output>
void update_block(std::size_t i, std::size_t k, D norm2, const Output& out, const Objective& obj,
                  D* block, D* scores, int) {
  sdca::update_dual_variables(i, k, norm2, out, obj, block, scores);
}

}  // namespace

#endif

// Minimal dense-feature training and a score-gradient call.
#include <iostream>
#include <vector>
#include "sdca/solver.h"
#include "sdca/loss.h"
// Include once in the executable, or compile this implementation separately.
#include "sdca/utility/logging.cpp"

int main() {
  // Column-major features: dimensions x examples. Three identity examples.
  double features[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double weights[9] = {}, dual[9] = {};
  std::vector<sdca::size_type> labels = {0, 1, 2};
  sdca::logging::set_level(sdca::logging::level::none);
  auto context = sdca::make_context(
      sdca::make_input_feature(3, 3, features), sdca::multiclass_output(3, labels),
      sdca::l2_hinge_topk_smooth<double, double>(1., 2, .7), dual, weights);
  context.criteria.eval_epoch = 1;
  context.criteria.epsilon = 1e-8;
  sdca::make_solver(context).solve();
  double scores[] = {.3, -.8, 1.2}, gradient[3];
  // In the core API, positive labels occupy the first p entries.
  double value =
      sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_alpha, 3, 1, scores, gradient, 2, .7);
  std::cout << context.status_name() << " loss=" << value << " gradient=" << gradient[0] << ","
            << gradient[1] << "," << gradient[2] << "\n";
  return context.status == sdca::solver_status::solved ? 0 : 1;
}

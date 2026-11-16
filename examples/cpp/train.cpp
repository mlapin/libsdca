// Minimal dense-feature training of a shared-name objective and a row loss.
#include <iostream>
#include <utility>
#include <vector>
#include "sdca/loss.h"
#include "sdca/solver.h"

// Trains the objective it is given on three identity examples.
struct train {
  sdca::solver_status& status;
  template <typename Objective>
  void operator()(Objective objective) const {
    // Features (examples, dimensions), row-major; weights (classes, dimensions), dual
    // (examples, classes).
    double features[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double weights[9] = {}, dual[9] = {};
    std::vector<int> labels = {0, 1, 2};
    auto context = sdca::make_context(sdca::make_input_feature(3, 3, features),
                                      sdca::make_output_multiclass(labels.begin(), labels.end(), 3),
                                      std::move(objective), weights, dual);
    context.criteria.eval_epoch = 1;
    context.criteria.tol = 1e-8;
    sdca::make_solver(context).solve();
    status = context.status;
  }
};

int main() {
  // The smoothed top-2 hinge (alpha variant): l2_topk_hinge_alpha_smooth<double, double>.
  sdca::solver_status status = sdca::solver_status::failed;
  sdca::visit_multiclass_objective<double, double>(sdca::topk_hinge_alpha_objective(1., 2, .7),
                                                   train{status});

  // The loss of scores in class order at label 1, and its gradient.
  double scores[] = {.3, -.8, 1.2}, gradient[3];
  sdca::loss_parameters params;
  params.k = 2;
  params.gamma = .7;
  const double value =
      sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_alpha, 3, scores, 1, gradient, params);
  std::cout << sdca::solver_status_name(status) << " loss=" << value << " gradient=" << gradient[0]
            << "," << gradient[1] << "," << gradient[2] << "\n";
  return status == sdca::solver_status::solved ? 0 : 1;
}

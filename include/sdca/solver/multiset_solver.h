#ifndef SDCA_SOLVER_MULTISET_SOLVER_H
#define SDCA_SOLVER_MULTISET_SOLVER_H

#include "sdca/solver/base_solver.h"
#include "sdca/solver/context.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Dataset>
class multiset_solver
    : public base_solver<Result> {
public:
  typedef base_solver<Result> base;

  typedef Data data_type;
  typedef Result result_type;
  typedef Dataset dataset_type;

  typedef solver_context<Data, Dataset> context_type;
  typedef test_point<Result> evaluation_type;
  typedef std::vector<evaluation_type> evaluations_type;


  explicit multiset_solver(
      context_type&& __ctx
    ) :
      base::base_solver(
        __ctx.criteria,
        __ctx.test[0].in.num_examples,
        __ctx.test[0].out.num_classes
      ),
      context_(std::move(__ctx))
  {
    scores_.resize(base::num_classes_);
    evals_.resize(context_.test.size());
  }


  const std::vector<evaluations_type>& evaluations() const { return evals_; }


protected:
  context_type context_;
  std::vector<data_type> scores_;
  std::vector<evaluations_type> evals_;


  void evaluate_solution() override {
    // First, evaluate the current solution on the training data
    evals_[0].push_back(evaluate_train());
    log_eval(0, evals_[0].back());

    // Next, evaluate on any other (test) data sets
    for (size_type i = 1; i < evals_.size(); ++i) {
      evals_[i].push_back(evaluate_test(context_.test[i]));
      log_eval(i, evals_[i].back());
    }
  }


  inline void
  log_eval(size_type id, evaluation_type& eval) {
    LOG_VERBOSE << "  "
      "dataset " << id + 1 << ": "
      << eval.to_string() <<
      "eval_wall_time = " << this->eval_wall_timer_.elapsed_now() << ", "
      "eval_cpu_time = " << this->eval_cpu_timer_.elapsed_now() << std::endl;
  }


  inline void
  swap_ground_truth(
      const size_type label
    ) {
    // Swap the ground truth label and a label at 0
    std::swap(scores_[0], scores_[label]);
  }


  inline void
  swap_ground_truth(
      const size_type label,
      data_type* variables
    ) {
    // Swap the ground truth label and a label at 0
    std::swap(variables[0], variables[label]);
    std::swap(scores_[0], scores_[label]);
  }


  virtual evaluation_type
  evaluate_train() = 0;


  virtual evaluation_type
  evaluate_test(const dataset_type& set) = 0;

};

}

#endif

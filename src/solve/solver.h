#ifndef SDCA_SOLVE_SOLVER_H
#define SDCA_SOLVE_SOLVER_H

#include <algorithm>
#include <limits>
#include <random>

#include "util/logging.h"
#include "util/stopwatch.h"
#include "solvedef.h"

namespace sdca {

static const char* solver_status_name[] = {
  "none",
  "solving",
  "solved",
  "no_progress",
  "max_epoch",
  "max_cpu_time",
  "max_wall_time",
  "failed"
};

template <typename Result>
class solver_base {
public:
  typedef Result result_type;
  typedef train_point<result_type> record_type;
  static constexpr result_type sufficient_increase = 1
    - 16 * std::numeric_limits<result_type>::epsilon();

  solver_base(
      const stopping_criteria& __criteria,
      const size_type __num_examples
    ) :
      criteria_(__criteria),
      num_examples_(__num_examples),
      status_(solver_status::none),
      epoch_(0),
      primal_loss_(0),
      dual_loss_(0),
      regularizer_(0),
      primal_(0),
      dual_(0),
      gap_(0)
  {}

  constexpr const char*
  name() const { return "stochastic dual coordinate ascent"; }

  void solve() {
    initialize();
    begin_solve();
    while (status_ == solver_status::solving) {
      begin_epoch();
      for (size_type i = 0; i < num_examples_; ++i) {
        solve_example(examples_[i]);
      }
      end_epoch();
    }
    end_solve();
  }

  solver_status status() const { return status_; }

  std::string status_name() const {
    return solver_status_name[static_cast<solver_status_type>(status_)];
  }

  size_type epoch() const { return epoch_; }

  double solve_cpu_time() const {
    return solve_cpu_.elapsed;
  }

  double solve_wall_time() const {
    return solve_wall_.elapsed;
  }

  double eval_cpu_time() const {
    return eval_cpu_.elapsed;
  }

  double eval_wall_time() const {
    return eval_wall_.elapsed;
  }

  double cpu_time() const {
    return solve_cpu_.elapsed + eval_cpu_.elapsed;
  }

  double wall_time() const {
    return solve_wall_.elapsed + eval_wall_.elapsed;
  }

  const result_type primal() const { return primal_; }

  const result_type dual() const { return dual_; }

  const result_type absolute_gap() const { return gap_; }

  const result_type relative_gap() const {
    result_type max = std::max(std::abs(primal_), std::abs(dual_));
    return (max > static_cast<result_type>(0))
      ? (max < std::numeric_limits<result_type>::infinity()
        ? gap_ / max
        : std::numeric_limits<result_type>::infinity())
      : static_cast<result_type>(0);
  }

  const std::vector<record_type>& records() const { return records_; }

protected:
  const stopping_criteria criteria_;
  const size_type num_examples_;

  // Current progress
  solver_status status_;
  stopwatch_cpu solve_cpu_;
  stopwatch_wall solve_wall_;
  stopwatch_cpu eval_cpu_;
  stopwatch_wall eval_wall_;
  size_type epoch_;
  result_type primal_loss_;
  result_type dual_loss_;
  result_type regularizer_;
  result_type primal_;
  result_type dual_;
  result_type gap_;

  // Other
  bool recompute_gap_;
  std::minstd_rand generator_;
  std::vector<size_type> examples_;
  std::vector<record_type> records_;

  // Initialization
  virtual void initialize() {
    status_ = (criteria_.max_epoch > 0)
      ? solver_status::solving
      : solver_status::max_epoch;
    epoch_ = 0;

    solve_cpu_.start();
    solve_wall_.start();
    eval_cpu_.reset();
    eval_wall_.reset();

    primal_loss_ = 0;
    dual_loss_ = 0;
    regularizer_ = 0;
    primal_ = std::numeric_limits<result_type>::infinity();
    dual_ = -std::numeric_limits<result_type>::infinity();
    gap_ = std::numeric_limits<result_type>::infinity();

    recompute_gap_ = false;
    generator_.seed();
    examples_.resize(num_examples_);
    std::iota(examples_.begin(), examples_.end(), 0);
  }

  virtual void begin_solve() {
    if (criteria_.check_on_start) {
      compute_duality_gap();
    }
  }

  virtual void end_solve() {
    if (recompute_gap_) {
      compute_duality_gap();
    }
    solve_cpu_.stop();
    solve_wall_.stop();
    LOG_INFO << "status: " << status_name() << " ("
      "epoch = " << epoch() << ", "
      "relative_gap = " << relative_gap() << ", "
      "solve_time: " << solve_wall_.elapsed << ", "
      "eval_time: " << eval_wall_.elapsed << ", "
      "wall_time: " << wall_time() << ", "
      "cpu_time: " << cpu_time() << ")" << std::endl;
  }

  virtual void begin_epoch() {
    recompute_gap_ = true;
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }

  virtual void end_epoch() {
    ++epoch_;
    solve_cpu_.stop();
    solve_wall_.stop();
    if ((criteria_.check_epoch > 0) && (epoch_ % criteria_.check_epoch == 0)) {
      compute_duality_gap();
    } else {
      LOG_VERBOSE << "  "
        "epoch: " << std::setw(3) << epoch() << std::setw(0) << ", "
        "solve_time: " << solve_wall_.elapsed << ", "
        "eval_time: " << eval_wall_.elapsed << ", "
        "wall_time: " << wall_time() << ", "
        "cpu_time: " << cpu_time() << std::endl;
    }
    if (status_ == solver_status::solving) {
      if (epoch() >= criteria_.max_epoch) {
        status_ = solver_status::max_epoch;
        LOG_DEBUG << "  (warning) "
          "epoch limit: " << epoch() << std::endl;
      } else if (criteria_.max_cpu_time > 0 &&
          cpu_time() >= criteria_.max_cpu_time) {
        status_ = solver_status::max_cpu_time;
        LOG_DEBUG << "  (warning) "
          "cpu time limit: " << cpu_time() << std::endl;
      } else if (criteria_.max_wall_time > 0 &&
          wall_time() >= criteria_.max_wall_time) {
        status_ = solver_status::max_wall_time;
        LOG_DEBUG << "  (warning) "
          "wall time limit: " << wall_time() << std::endl;
      }
    }
    solve_cpu_.resume();
    solve_wall_.resume();
  }

  virtual void compute_duality_gap() {
    recompute_gap_ = false;
    result_type dual_before = dual_;
    eval_cpu_.resume();
    eval_wall_.resume();
    evaluate_solution();
    eval_cpu_.stop();
    eval_wall_.stop();
    result_type max = std::max(std::abs(primal_), std::abs(dual_));
    if (gap_ <= max * static_cast<result_type>(criteria_.epsilon)) {
      status_ = solver_status::solved;
      if (gap_ < - max * std::numeric_limits<result_type>::epsilon()) {
        status_ = solver_status::failed;
        LOG_DEBUG << "  (warning) "
          "failed due to negative duality gap: " << gap_ << std::endl;
      }
    } else if (dual_ < sufficient_increase * dual_before) {
      status_ = solver_status::no_progress;
      LOG_DEBUG << "  (warning) "
        "no progress due to insufficient dual objective increase: "
        << (dual_ - dual_before) << std::endl;
    }
    records_.emplace_back(
      primal_, dual_, gap_, primal_loss_, dual_loss_, regularizer_,
      epoch_, cpu_time(), wall_time(), solve_cpu_.elapsed, solve_wall_.elapsed,
      eval_cpu_.elapsed, eval_wall_.elapsed);
    LOG_VERBOSE << "  "
      "epoch: " << std::setw(3) << epoch() << std::setw(0) << ", "
      "primal: " << primal_ << ", "
      "dual: " << dual_ << ", "
      "absolute_gap: " << gap_ << ", "
      "relative_gap: " << relative_gap() << ", "
      "solve_time: " << solve_wall_.elapsed << ", "
      "eval_time: " << eval_wall_.elapsed << ", "
      "wall_time: " << wall_time() << ", "
      "cpu_time: " << cpu_time() << std::endl;
  }

  virtual void solve_example(const size_type i) = 0;

  virtual void evaluate_solution() = 0;

};

template <typename Data,
          typename Result>
class multiset_solver : public solver_base<Result> {
public:
  typedef solver_base<Result> base;
  typedef solver_context<Data> context_type;
  typedef dataset<Data> dataset_type;
  typedef test_point<Result> evaluation_type;
  typedef std::vector<evaluation_type> evaluations_type;

  multiset_solver(
      const context_type& __ctx
    ) :
      base::solver_base(__ctx.criteria, __ctx.datasets[0].num_examples),
      context_(__ctx),
      evals_(__ctx.datasets.size())
  {}

  const std::vector<evaluations_type>& evaluations() const { return evals_; }

protected:
  const context_type& context_;
  std::vector<evaluations_type> evals_;

  void evaluate_solution() override {
    auto datasets = context_.datasets;
    evals_[0].push_back(evaluate_train());
    log_eval(0, evals_[0].back());
    for (size_type i = 1; i < evals_.size(); ++i) {
      evals_[i].push_back(evaluate_test(datasets[i]));
      log_eval(i, evals_[i].back());
    }
  }

  inline void
  log_eval(size_type id, evaluation_type& eval) {
    LOG_VERBOSE << "  "
      "eval " << id + 1 << ": "
      << eval.to_string() <<
      "wall_time = " << this->eval_wall_.elapsed_now() << ", "
      "cpu_time = " << this->eval_cpu_.elapsed_now() << std::endl;
  }

  virtual evaluation_type
  evaluate_train() = 0;

  virtual evaluation_type
  evaluate_test(const dataset_type& set) = 0;
};

}

#endif

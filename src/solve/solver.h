#ifndef SDCA_SOLVE_SOLVER_H
#define SDCA_SOLVE_SOLVER_H

#include <algorithm>
#include <random>

#include "util/logging.h"
#include "solvedef.h"

namespace sdca {

template <typename Result>
class solver_base {
public:
  typedef Result result_type;
  typedef train_point<result_type> record_type;
  static constexpr result_type sufficient_increase = static_cast<Result>(1)
    + std::numeric_limits<result_type>::epsilon();

  solver_base(
      const stopping_criteria& __criteria,
      const size_type __num_examples
    ) :
      criteria_(__criteria),
      num_examples_(__num_examples),
      status_(solver_status::none),
      epoch_(0),
      cpu_time_(0),
      wall_time_(0),
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

  double cpu_time() const { return cpu_time_; }

  double wall_time() const { return wall_time_; }

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
  size_type epoch_;
  cpu_time_point cpu_start_;
  wall_time_point wall_start_;
  double cpu_time_;
  double wall_time_;
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
    cpu_start_ = std::clock();
    wall_start_ = wall_clock::now();

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
    cpu_time_ += cpu_time_now();
    wall_time_ += wall_time_now();
    LOG_INFO << "status: " << status_name() << " ("
      "epoch = " << epoch() << ", "
      "relative_gap = " << relative_gap() << ", "
      "cpu_time = " << cpu_time_now() << ", "
      "wall_time = " << wall_time_now() << ")" << std::endl;
  }

  virtual void begin_epoch() {
    recompute_gap_ = true;
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }

  virtual void end_epoch() {
    ++epoch_;
    if ((criteria_.check_epoch > 0) && (epoch_ % criteria_.check_epoch == 0)) {
      compute_duality_gap();
    }
    if (status_ == solver_status::solving) {
      if (epoch() >= criteria_.max_epoch) {
        status_ = solver_status::max_epoch;
        LOG_DEBUG << "  (warning) "
          "epoch limit: " << epoch() << std::endl;
      } else if (criteria_.max_cpu_time > 0 &&
          cpu_time_now() >= criteria_.max_cpu_time) {
        status_ = solver_status::max_cpu_time;
        LOG_DEBUG << "  (warning) "
          "cpu time limit: " << cpu_time_now() << std::endl;
      } else if (criteria_.max_wall_time > 0 &&
          wall_time_now() >= criteria_.max_wall_time) {
        status_ = solver_status::max_wall_time;
        LOG_DEBUG << "  (warning) "
          "wall time limit: " << wall_time_now() << std::endl;
      }
    }
  }

  virtual void compute_duality_gap() {
    recompute_gap_ = false;
    result_type dual_before = dual_;
    evaluate_solution();
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
      epoch_, cpu_time_now(), wall_time_now());
    LOG_VERBOSE << "  "
      "epoch: " << std::setw(3) << epoch() << std::setw(0) << ", "
      "primal: " << primal() << ", "
      "dual: " << dual() << ", "
      "absolute_gap: " << absolute_gap() << ", "
      "relative_gap: " << relative_gap() << ", "
      "cpu_time: " << cpu_time_now() << ", "
      "wall_time: " << wall_time_now() << std::endl;
  }

  double cpu_time_now() const {
    return static_cast<double>(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
  }

  double wall_time_now() const {
    return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - wall_start_).count();
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
      "cpu_time = " << this->cpu_time_now() << ", "
      "wall_time = " << this->wall_time_now() << std::endl;
  }

  virtual evaluation_type
  evaluate_train() = 0;

  virtual evaluation_type
  evaluate_test(const dataset_type& set) = 0;
};

}

#endif

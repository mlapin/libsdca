#ifndef SDCA_BASE_SOLVER_SOLVER_H
#define SDCA_BASE_SOLVER_SOLVER_H

#include <algorithm>
#include <limits>
#include <random>

#include "sdca/solver/solverdef.h"
#include "sdca/util/logging.h"
#include "sdca/util/stopwatch.h"

namespace sdca {

template <typename Result>
class base_solver {
public:
  typedef Result result_type;
  typedef train_point<Result> record_type;

  static constexpr Result sufficient_increase =
      1 - 16 * std::numeric_limits<Result>::epsilon();


  base_solver(
      const stopping_criteria& __criteria,
      const size_type __num_examples,
      const size_type __num_classes
    ) :
      criteria_(__criteria),
      num_examples_(__num_examples),
      num_classes_(__num_classes),
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

      for (auto& example : examples_) {
        solve_example(example);
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
    return solve_cpu_timer_.elapsed;
  }

  double solve_wall_time() const {
    return solve_wall_timer_.elapsed;
  }

  double eval_cpu_time() const {
    return eval_cpu_timer_.elapsed;
  }

  double eval_wall_time() const {
    return eval_wall_timer_.elapsed;
  }

  double cpu_time() const {
    return solve_cpu_timer_.elapsed + eval_cpu_timer_.elapsed;
  }

  double wall_time() const {
    return solve_wall_timer_.elapsed + eval_wall_timer_.elapsed;
  }

  const Result primal() const { return primal_; }

  const Result dual() const { return dual_; }

  const Result absolute_gap() const { return gap_; }

  const Result relative_gap() const {
    Result max = std::max(std::abs(primal_), std::abs(dual_));
    return (max > static_cast<Result>(0))
      ? (max < std::numeric_limits<Result>::infinity()
        ? gap_ / max
        : std::numeric_limits<Result>::infinity())
      : static_cast<Result>(0);
  }

  const std::vector<record_type>& records() const { return records_; }

protected:
  const stopping_criteria criteria_;
  const size_type num_examples_;
  const size_type num_classes_;

  // Current progress
  solver_status status_;
  stopwatch_cpu solve_cpu_timer_;
  stopwatch_wall solve_wall_timer_;
  stopwatch_cpu eval_cpu_timer_;
  stopwatch_wall eval_wall_timer_;
  size_type epoch_;
  Result primal_loss_;
  Result dual_loss_;
  Result regularizer_;
  Result primal_;
  Result dual_;
  Result gap_;

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

    solve_cpu_timer_.start();
    solve_wall_timer_.start();
    eval_cpu_timer_.reset();
    eval_wall_timer_.reset();

    primal_loss_ = 0;
    dual_loss_ = 0;
    regularizer_ = 0;
    primal_ = std::numeric_limits<Result>::infinity();
    dual_ = -std::numeric_limits<Result>::infinity();
    gap_ = std::numeric_limits<Result>::infinity();

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

    solve_cpu_timer_.stop();
    solve_wall_timer_.stop();

    LOG_INFO << "status: " << status_name() << " ("
      "epoch = " << epoch() << ", "
      "relative_gap = " << relative_gap() << ", "
      "solve_wall_time: " << solve_wall_timer_.elapsed << ", "
      "eval_wall_time: " << eval_wall_timer_.elapsed << ", "
      "wall_time: " << wall_time() << ", "
      "cpu_time: " << cpu_time() << ")" << std::endl;
  }


  virtual void begin_epoch() {
    recompute_gap_ = true;
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }


  virtual void end_epoch() {
    ++epoch_;
    solve_cpu_timer_.stop();
    solve_wall_timer_.stop();

    // Check the duality gap or log progress
    if ((criteria_.check_epoch > 0) &&
        (epoch_ % criteria_.check_epoch == 0)) {
      compute_duality_gap();
    } else {
      LOG_DEBUG << "  "
        "epoch: " << std::setw(3) << epoch() << std::setw(0) << ", "
        "solve_wall_time: " << solve_wall_timer_.elapsed << ", "
        "eval_wall_time: " << eval_wall_timer_.elapsed << ", "
        "wall_time: " << wall_time() << ", "
        "cpu_time: " << cpu_time() << std::endl;
    }

    // Check stopping conditions
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

    // Resume timers
    solve_cpu_timer_.resume();
    solve_wall_timer_.resume();
  }


  virtual void compute_duality_gap() {
    // Cache the previous value
    Result dual_before = dual_;

    // Evaluate the current solution and time the evaluation process
    eval_cpu_timer_.resume();
    eval_wall_timer_.resume();
    evaluate_solution();
    eval_cpu_timer_.stop();
    eval_wall_timer_.stop();

    // The current gap value is recent and need not be recomputed
    recompute_gap_ = false;

    // Check the stopping conditions (the duality gap)
    Result max = std::max(std::abs(primal_), std::abs(dual_));
    if (gap_ <= max * static_cast<Result>(criteria_.epsilon)) {
      status_ = solver_status::solved;

      // The duality gap should be non-negative,
      // but can become negative due to roundoff errors.
      // However, a large negative gap indicates an issue in the code.
      if (gap_ < - max * std::numeric_limits<Result>::epsilon()) {
        status_ = solver_status::failed;
        LOG_DEBUG << "  (warning) "
          "failed due to negative duality gap: " << gap_ << std::endl;
      }

    } else if (dual_ < sufficient_increase * dual_before) {
      // The dual objective should only increase,
      // but it can (slightly) decrease due to roundoff errors,
      // especially when it is close to the saturation point.
      // A small decrease may indicate that
      // the solver cannot make further progress due to
      // the limitations of the floating-point arithmetic
      // (e.g. when the problem is ill-conditioned).
      // However, a large decrease may indicate an issue in the code.
      status_ = solver_status::no_progress;
      LOG_DEBUG << "  (warning) "
        "no progress due to insufficient dual objective increase: "
        << (dual_ - dual_before) << std::endl;
    }

    // Create a record of the current state
    records_.emplace_back(
      primal_, dual_, gap_,
      primal_loss_, dual_loss_, regularizer_,
      epoch_, cpu_time(), wall_time(),
      solve_cpu_timer_.elapsed, solve_wall_timer_.elapsed,
      eval_cpu_timer_.elapsed, eval_wall_timer_.elapsed
      );

    LOG_VERBOSE << "  "
      "epoch: " << std::setw(3) << epoch() << std::setw(0) << ", "
      "primal: " << primal_ << ", "
      "dual: " << dual_ << ", "
      "absolute_gap: " << gap_ << ", "
      "relative_gap: " << relative_gap() << ", "
      "solve_wall_time: " << solve_wall_timer_.elapsed << ", "
      "eval_wall_time: " << eval_wall_timer_.elapsed << ", "
      "wall_time: " << wall_time() << ", "
      "cpu_time: " << cpu_time() << std::endl;
  }


  virtual void
  solve_example(const size_type i) = 0;


  virtual void
  evaluate_solution() = 0;

};

}

#endif

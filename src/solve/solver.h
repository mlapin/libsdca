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

  const std::vector<state<result_type>>& states() const { return states_; }

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
  result_type primal_;
  result_type dual_;
  result_type gap_;
  std::vector<state<result_type>> states_;

  // Other
  bool recompute_gap_;
  std::minstd_rand generator_;
  std::vector<size_type> examples_;

  // Initialization
  virtual void initialize() {
    status_ = solver_status::solving;
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
      if (criteria_.max_epoch > 0 &&
          epoch() >= criteria_.max_epoch) {
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
    compute_objectives();
    result_type max = std::max(std::abs(primal_), std::abs(dual_));
    if (gap_ <= static_cast<result_type>(criteria_.epsilon) * max) {
      status_ = (gap_ > -std::numeric_limits<result_type>::epsilon())
        ? solver_status::solved : solver_status::failed;
    } else if (dual_ < sufficient_increase * dual_before) {
      status_ = solver_status::no_progress;
      LOG_DEBUG << "  (warning) "
        "no progress due to insufficient dual objective increase: "
        << (dual_ - dual_before) << std::endl;
    }
    states_.emplace_back(
      epoch(), cpu_time_now(), wall_time_now(), primal_, dual_, gap_);
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

  virtual void compute_objectives() = 0;

};

}

#endif

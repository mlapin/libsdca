#ifndef SDCA_SOLVE_SOLVER_H
#define SDCA_SOLVE_SOLVER_H

#include <algorithm>
#include <random>

#include "logging/logging.h"
#include "solvedef.h"

namespace sdca {

template <typename Result = long double>
class solver_base {
public:
  typedef Result result_type;
  static constexpr result_type dual_decrease_tolerance =
    1.0 + 8.0 * std::numeric_limits<double>::epsilon();

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
  {
    LOG_INFO << "stopping_criteria: "
      "epsilon = " << __criteria.epsilon << ", "
      "check_epoch = " << __criteria.check_epoch << ", "
      "max_num_epoch = " << __criteria.max_num_epoch << ", "
      "max_cpu_time = " << __criteria.max_cpu_time << ", "
      "max_wall_time = " << __criteria.max_wall_time << std::endl;
  }

  void solve() {
    begin_solve();
    for (epoch_ = 0; epoch_ < criteria_.max_num_epoch; ++epoch_) {
      begin_epoch();
      for (size_type i = 0; i < num_examples_; ++i) {
        solve_example(examples_[i]);
      }
      if (end_epoch()) {
        break;
      }
    }
    end_solve();
  }

  solver_status status() const { return status_; }

  std::string status_name() const {
    return solver_status_name[static_cast<solver_status_type>(status_)];
  }

  size_type num_epoch() const { return epoch_ + 1; }

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


  virtual void begin_solve() {
    cpu_start_ = std::clock();
    wall_start_ = wall_clock::now();

    status_ = solver_status::solving;
    primal_ = std::numeric_limits<result_type>::infinity();
    dual_ = -std::numeric_limits<result_type>::infinity();
    gap_ = std::numeric_limits<result_type>::infinity();
    recompute_gap_ = false;

    generator_.seed();
    examples_.resize(num_examples_);
    std::iota(examples_.begin(), examples_.end(), 0);
  }

  virtual void end_solve() {
    if (status_ == solver_status::solving &&
        epoch_ >= criteria_.max_num_epoch) {
      status_ = solver_status::max_num_epoch;
      if (epoch_ > 0) {
        --epoch_; // correct to the last executed epoch
      }
      LOG_DEBUG << "  (warning) "
        "epochs limit: " << num_epoch() << std::endl;
    }
    if (recompute_gap_) {
      compute_duality_gap();
    }
    cpu_time_ += cpu_time_now();
    wall_time_ += wall_time_now();
    LOG_INFO << "status: " << status_name() << " ("
      "epoch = " << num_epoch() << ", "
      "relative_gap = " << relative_gap() << ", "
      "cpu_time = " << cpu_time_now() << ", "
      "wall_time = " << wall_time_now() << ")" << std::endl;
  }

  virtual void begin_epoch() {
    recompute_gap_ = true;
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }

  virtual bool end_epoch() {
    if ((criteria_.check_epoch > 0) &&
        ((epoch_ % criteria_.check_epoch) == (criteria_.check_epoch - 1))) {
      compute_duality_gap();
    }
    if (status_ == solver_status::solving) {
      if (criteria_.max_cpu_time > 0 &&
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
    return status_ != solver_status::solving;
  }

  virtual void compute_duality_gap() {
    recompute_gap_ = false;
    result_type dual_before = dual_;
    compute_objectives();
    result_type max = std::max(std::abs(primal_), std::abs(dual_));
    if (gap_ <= static_cast<result_type>(criteria_.epsilon) * max) {
      status_ = solver_status::solved;
    } else if (dual_ * dual_decrease_tolerance < dual_before) {
      status_ = solver_status::dual_decreased;
      LOG_DEBUG << "  (warning) "
        "dual objective decreased by " << (dual_before - dual_) << std::endl;
    }
    states_.emplace_back(
      num_epoch(), cpu_time_now(), wall_time_now(), primal_, dual_, gap_);
    LOG_VERBOSE << "  "
      "epoch: " << std::setw(3) << num_epoch() << std::setw(0) << ", "
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

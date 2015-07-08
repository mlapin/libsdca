#ifndef SDCA_SOLVE_SOLVER_H
#define SDCA_SOLVE_SOLVER_H

#include <algorithm>
#include <random>

#include "solvedef.h"

namespace sdca {

template <typename real_type>
class solver {
public:
  solver(
      const std::size_t __num_examples,
      const sdca::stopping_criteria& __criteria
    ) :
      num_examples_(__num_examples),
      criteria_(__criteria),
      status_(sdca::status::none),
      epoch_(0),
      cpu_start_(0),
      cpu_end_(0),
      primal_(0),
      dual_(0),
      recompute_duality_gap_(false)
  {}

  void solve() {
    begin_solve();
    for (epoch_ = 0; epoch_ < criteria_.max_num_epoch; ++epoch_) {
      begin_epoch();
      for (std::size_t i = 0; i < num_examples_; ++i) {
        solve_example(examples_[i]);
      }
      if (end_epoch()) {
        break;
      }
    }
    end_solve();
  }

  const std::size_t num_examples() const { return num_examples_; }

  const sdca::stopping_criteria& stopping_criteria() const { return criteria_; }

  const sdca::status status() const { return status_; }

  const std::string status_name() const { return sdca::status_name[status_]; }

  const std::size_t epoch() const { return epoch_; }

  const double cpu_time() const {
    return static_cast<double>(cpu_end_ - cpu_start_) / CLOCKS_PER_SEC;
  }

  const double wall_time() const {
    return std::chrono::duration<double>(wall_end_ - wall_start_).count();
  }

  const double cpu_time_now() const {
    return static_cast<double>(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
  }

  const double wall_time_now() const {
    return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - wall_start_).count();
  }

  const real_type primal() const { return primal_; }

  const real_type dual() const { return dual_; }

  const real_type absolute_gap() const { return primal() - dual(); }

  const real_type relative_gap() const {
    real_type max = std::max(std::abs(primal()), std::abs(dual()));
    return (max > static_cast<real_type>(0))
      ? (max < std::numeric_limits<real_type>::infinity()
        ? absolute_gap() / max
        : std::numeric_limits<real_type>::infinity())
      : static_cast<real_type>(0);
  }

  const std::vector<state<real_type>>& states() const { return states_; }

protected:
  const std::size_t num_examples_;
  const sdca::stopping_criteria criteria_;

  // Current progress
  sdca::status status_;
  std::size_t epoch_;
  cpu_time_point cpu_start_;
  cpu_time_point cpu_end_;
  wall_time_point wall_start_;
  wall_time_point wall_end_;
  real_type primal_;
  real_type dual_;
  std::vector<state<real_type>> states_;

  // Other
  bool recompute_duality_gap_;
  std::minstd_rand generator_;
  std::vector<std::size_t> examples_;


  virtual void begin_solve() {
    cpu_start_ = std::clock();
    wall_start_ = wall_clock::now();
    cpu_end_ = cpu_start_;
    wall_end_ = wall_start_;

    status_ = sdca::status::solving;
    primal_ = std::numeric_limits<real_type>::infinity();
    dual_ = -std::numeric_limits<real_type>::infinity();
    recompute_duality_gap_ = false;

    generator_.seed();
    examples_.resize(num_examples_);
    std::iota(examples_.begin(), examples_.end(), 0);
  }

  virtual void end_solve() {
    if (status_ == sdca::status::solving &&
        epoch_ >= criteria_.max_num_epoch) {
      status_ = sdca::status::max_num_epoch;
      if (epoch_ > 0) {
        --epoch_; // correct to the last executed epoch
      }
    }
    if (recompute_duality_gap_) {
      compute_duality_gap();
    }
    cpu_end_ = std::clock();
    wall_end_ = wall_clock::now();
  }

  virtual void begin_epoch() {
    recompute_duality_gap_ = true;
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }

  virtual bool end_epoch() {
    if ((criteria_.check_epoch > 0) &&
        ((epoch_ % criteria_.check_epoch) == (criteria_.check_epoch - 1))) {
      compute_duality_gap();
    }
    if (status_ == sdca::status::solving) {
      if (criteria_.max_cpu_time > 0 &&
          cpu_time_now() >= criteria_.max_cpu_time) {
        status_ = sdca::status::max_cpu_time;
      } else if (criteria_.max_wall_time > 0 &&
          wall_time_now() >= criteria_.max_wall_time) {
        status_ = sdca::status::max_wall_time;
      }
    }
    return status_ != sdca::status::solving;
  }

  virtual void compute_duality_gap() {
    recompute_duality_gap_ = false;
    real_type before = dual();
    compute_objectives();
    if (relative_gap() <= criteria_.epsilon) {
      status_ = sdca::status::solved;
    } else {
      real_type after = dual() * (static_cast<real_type>(1)
        + std::numeric_limits<real_type>::epsilon());
      if (after < before) {
        status_ = sdca::status::dual_decreased;
      }
    }
    states_.emplace_back(
      epoch_, cpu_time_now(), wall_time_now(), primal(), dual());
  }

  virtual void solve_example(std::size_t example) = 0;

  virtual void compute_objectives() = 0;

};

}

#endif

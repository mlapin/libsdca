#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <random>

#include "solver.hpp"

namespace sdca {

template <typename RealType>
void Solver<RealType>::Solve() {

  BeginSolve();

  for (epoch_ = 0; epoch_ < max_num_epoch_; ++epoch_) {

    BeginEpoch();

    for (SizeType i = 0; i < num_examples_; ++i) {
      SolveExample(examples_[i]);
    }

    if (EndEpoch()) {
      break;
    }

  }

  EndSolve();

}


template <typename RealType>
void Solver<RealType>::BeginSolve() {
#ifdef VERBOSE_BEGIN_END
  std::cout << get_solver_name() << "::BeginSolve(" <<
    std::scientific << std::setprecision(16) <<
    "num_examples: " << get_num_examples() << ", "
    "num_tasks: " << get_num_tasks() << ", "
    "check_gap_frequency: " << get_check_gap_frequency() << ", "
    "max_num_epoch: " << get_max_num_epoch() << ", "
    "seed: " << get_seed() << ", "
    "epsilon: " << get_epsilon() << ")" << std::endl;
#endif

  cpu_start_ = std::clock();
  wall_start_ = std::chrono::high_resolution_clock::now();
  cpu_end_ = cpu_start_;
  wall_end_ = wall_start_;

  status_ = Status::Solving;
  recompute_duality_gap_ = false;
  primal_objective_ = std::numeric_limits<RealType>::infinity();
  dual_objective_ = -std::numeric_limits<RealType>::infinity();

  generator_.seed(static_cast<std::minstd_rand::result_type>(seed_));

  examples_.resize(num_examples_);
  std::iota(examples_.begin(), examples_.end(), static_cast<SizeType>(0));
}

template <typename RealType>
void Solver<RealType>::EndSolve() {
  if (status_ == Status::Solving && epoch_ >= max_num_epoch_) {
    status_ = Status::MaxNumEpoch;
    if (epoch_ > 0) {
      --epoch_; // correct to the last executed epoch
    }
  }

  if (recompute_duality_gap_) {
    ComputeDualityGap();
  }

  cpu_end_ = std::clock();
  wall_end_ = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_BEGIN_END
  std::cout << get_solver_name() << "::EndSolve(" <<
    "status: " << get_status_name() << ", "
    "epoch: " << get_num_epoch() << ", "
    "relative_gap: " << get_relative_gap() << ", "
    "absolute_gap: " << get_absolute_gap() << ", "
    "primal: " << get_primal_objective() << ", "
    "dual: " << get_dual_objective() << ", "
    "cpu_time: " << get_cpu_time() << ", "
    "wall_time: " << get_wall_time() << ")" << std::endl;
#endif
  std::cout.copyfmt(std::ios(nullptr));
}

template <typename RealType>
void Solver<RealType>::BeginEpoch() {
  recompute_duality_gap_ = true;
  std::shuffle(examples_.begin(), examples_.end(), generator_);
}

template <typename RealType>
bool Solver<RealType>::EndEpoch() {
  // Check duality gap every 'check_gap_frequency_' epoch
  bool check_now = (check_gap_frequency_ > 0)
    && (epoch_ % check_gap_frequency_) == (check_gap_frequency_ - 1);
  if (check_now) {
    ComputeDualityGap();
  }

  if (status_ == Status::Solving) {
    if (max_cpu_time_ > 0 && get_cpu_time_now() >= max_cpu_time_) {
      status_ = Status::MaxCpuTime;
    }
    if (max_wall_time_ > 0 && get_wall_time_now() >= max_wall_time_) {
      status_ = Status::MaxWallTime;
    }
  }

  // Stop if not solving
  return status_ != Status::Solving;
}

template <typename RealType>
void Solver<RealType>::ComputeDualityGap() {
  recompute_duality_gap_ = false;
  RealType before = get_dual_objective();

  ComputePrimalDualObjectives();

  if (get_relative_gap() <= epsilon_) {
    status_ = Status::Solved;
  } else {
    RealType after = get_dual_objective() * (static_cast<RealType>(1)
      + kInaccuracyTolerance);
    if (after < before) {
      status_ = Status::DualObjectiveDecreased;
    }
  }

#ifdef VERBOSE_DUALITY_GAP
  std::cout << "  "
    "epoch: " << std::setw(4) << get_num_epoch() << std::setw(0) << ", "
    "primal: " << get_primal_objective() << ", "
    "dual: " << get_dual_objective() << ", "
    "absolute_gap: " << get_absolute_gap() << ", "
    "relative_gap: " << get_relative_gap() << ", "
    "cpu_time: " << get_cpu_time_now() << ", "
    "wall_time: " << get_wall_time_now() << std::endl;
#endif

}


template class Solver<float>;
template class Solver<double>;

}

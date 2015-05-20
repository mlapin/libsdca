#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
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
  cpu_start_ = std::clock();
  wall_start_ = std::chrono::high_resolution_clock::now();
  cpu_end_ = cpu_start_;
  wall_end_ = wall_start_;

  status_ = Status::Solving;
  recompute_duality_gap_ = false;

  generator_.seed(seed_);

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
    RealType after = get_dual_objective();
    after += std::numeric_limits<RealType>::epsilon() * after;
    if (after < before) {
      status_ = Status::DualObjectiveDecreased;
    }
  }

}


template class Solver<float>;
template class Solver<double>;

}

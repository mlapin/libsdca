#ifndef SDCA_SOLVER_HPP
#define SDCA_SOLVER_HPP

#include <chrono>
#include <ctime>
#include <vector>
#include <random>

#include "common.hpp"

namespace sdca {

template <typename RealType = double>
class Solver {

enum class Status {
  Solving = -1,
  Solved = 0,
  DualObjectiveDecreased,
  MaxNumEpoch
};

using WallClock = std::chrono::high_resolution_clock;
using WallTimePoint = std::chrono::time_point<WallClock>;
using CpuTimePoint = std::clock_t;

public:
  static constexpr RealType kDefaultLambda = 1;
  static constexpr RealType kDefaultEpsilon = static_cast<RealType>(1e-2);
  static constexpr SizeType kDefaultMaxNumEpoch = 1000;
  static constexpr SizeType kDefaultCheckGapFrequency = 10;
  static constexpr SizeType kDefaultSeed = 1;

  Solver(
      const SizeType num_examples,
      const SizeType num_tasks,
      const RealType lambda = kDefaultLambda,
      const RealType epsilon = kDefaultEpsilon,
      const SizeType max_num_epoch = kDefaultMaxNumEpoch,
      const SizeType check_gap_frequency = kDefaultCheckGapFrequency,
      const SizeType seed = kDefaultSeed
    ) :
      num_examples_(num_examples),
      num_tasks_(num_tasks),
      lambda_(lambda),
      epsilon_(epsilon),
      max_num_epoch_(max_num_epoch),
      check_gap_frequency_(check_gap_frequency),
      seed_(seed),
      primal_objective_(0),
      dual_objective_(0),
      status_(Status::Solved),
      epoch_(0),
      cpu_start_(0),
      cpu_end_(0),
      recompute_duality_gap_(false)
  {}

  void Solve();

  SizeType get_num_examples() const { return num_examples_; }

  SizeType get_num_tasks() const { return num_tasks_; }

  RealType get_lambda() const { return lambda_; }
  void set_lambda(const RealType lambda) { lambda_ = lambda; }

  RealType get_epsilon() const { return epsilon_; }
  void set_epsilon(const RealType epsilon) { epsilon_ = epsilon; }

  SizeType get_max_num_epoch() const { return max_num_epoch_; }
  void set_max_num_epoch(const SizeType max_num_epoch) {
    max_num_epoch_ = max_num_epoch;
  }

  SizeType get_check_gap_frequency() const { return check_gap_frequency_; }
  void set_check_gap_frequency(const SizeType check_gap_frequency) {
    check_gap_frequency_ = check_gap_frequency;
  }

  SizeType get_seed() const { return seed_; }
  void set_seed(const SizeType seed) { seed_ = seed; }

  Status get_status() const { return status_; }

  SizeType get_epoch() const { return epoch_; }

  double get_cpu_time() const {
    return static_cast<double>(cpu_end_ - cpu_start_) / CLOCKS_PER_SEC;
  }

  double get_wall_time() const {
    return std::chrono::duration<double>(wall_end_ - wall_start_).count();
  }

  RealType get_primal_objective() const { return primal_objective_; }

  RealType get_dual_objective() const { return dual_objective_; }

  RealType get_absolute_gap() const {
    return get_primal_objective() - get_dual_objective();
  }

  RealType get_relative_gap() const {
    RealType max = std::max(
      std::abs(get_primal_objective()), std::abs(get_dual_objective()));
    return (max > static_cast<RealType>(0))
      ? get_absolute_gap() / max
      : static_cast<RealType>(0);
  }


protected:
  // Problem specification
  const SizeType num_examples_;
  const SizeType num_tasks_;

  // Solver parameters
  RealType lambda_;
  RealType epsilon_;
  SizeType max_num_epoch_;
  SizeType check_gap_frequency_;
  SizeType seed_;

  // Objectives
  RealType primal_objective_;
  RealType dual_objective_;

  // Current progress
  Status status_;
  SizeType epoch_;
  CpuTimePoint cpu_start_;
  CpuTimePoint cpu_end_;
  WallTimePoint wall_start_;
  WallTimePoint wall_end_;

  // Helper temporary variables
  bool recompute_duality_gap_;
  std::minstd_rand generator_;
  std::vector<SizeType> examples_;


  void BeginSolve();

  void EndSolve();

  void BeginEpoch();

  bool EndEpoch();

  void ComputeDualityGap();

  virtual void SolveExample(SizeType example) = 0;

  virtual void ComputePrimalDualObjectives() = 0;

};

}

#endif // SDCA_SOLVER_HPP

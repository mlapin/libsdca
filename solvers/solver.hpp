#ifndef SDCA_SOLVER_HPP
#define SDCA_SOLVER_HPP

#include <chrono>
#include <ctime>
#include <random>
#include <string>
#include <vector>

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
  static constexpr char const *kDefaultName = "Solver";
  static constexpr SizeType kDefaultCheckGapFrequency = 10;
  static constexpr SizeType kDefaultMaxNumEpoch = 100;
  static constexpr SizeType kDefaultSeed = 1;
  static constexpr RealType kDefaultEpsilon = static_cast<RealType>(1e-2);


  static constexpr RealType kInaccuracyTolerance =
    64 * std::numeric_limits<RealType>::epsilon();

  Solver(
      const SizeType num_examples,
      const SizeType num_tasks,
      const std::string solver_name = kDefaultName,
      const SizeType check_gap_frequency = kDefaultCheckGapFrequency,
      const SizeType max_num_epoch = kDefaultMaxNumEpoch,
      const SizeType seed = kDefaultSeed,
      const RealType epsilon = kDefaultEpsilon
    ) :
      num_examples_(num_examples),
      num_tasks_(num_tasks),
      solver_name_(solver_name),
      check_gap_frequency_(check_gap_frequency),
      max_num_epoch_(max_num_epoch),
      seed_(seed),
      epsilon_(epsilon),
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

  std::string get_solver_name() const { return solver_name_; }

  SizeType get_check_gap_frequency() const { return check_gap_frequency_; }
  void set_check_gap_frequency(const SizeType check_gap_frequency) {
    check_gap_frequency_ = check_gap_frequency;
  }

  SizeType get_max_num_epoch() const { return max_num_epoch_; }
  void set_max_num_epoch(const SizeType max_num_epoch) {
    max_num_epoch_ = max_num_epoch;
  }

  SizeType get_seed() const { return seed_; }
  void set_seed(const SizeType seed) { seed_ = seed; }

  RealType get_epsilon() const { return epsilon_; }
  void set_epsilon(const RealType epsilon) { epsilon_ = epsilon; }

  Status get_status() const { return status_; }

  std::string get_status_name() const {
    switch (status_) {
      case Status::Solved: return "Solved";
      case Status::Solving: return "Solving";
      case Status::MaxNumEpoch: return "MaxNumEpoch";
      case Status::DualObjectiveDecreased: return "DualObjectiveDecreased";
      default: return "Unknown";
    }
  }

  SizeType get_num_epoch() const { return epoch_ + 1; }

  double get_cpu_time() const {
    return static_cast<double>(cpu_end_ - cpu_start_) / CLOCKS_PER_SEC;
  }

  double get_wall_time() const {
    return std::chrono::duration<double>(wall_end_ - wall_start_).count();
  }

  double get_cpu_time_now() const {
    return static_cast<double>(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
  }

  double get_wall_time_now() const {
    return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - wall_start_).count();
  }

  RealType get_primal_objective() const {
    return primal_objective_;
  }

  RealType get_dual_objective() const {
    return dual_objective_;
  }

  RealType get_absolute_gap() const {
    return get_primal_objective() - get_dual_objective();
  }

  RealType get_relative_gap() const {
    RealType max = std::max(
      std::abs(get_primal_objective()), std::abs(get_dual_objective()));
    return (max > static_cast<RealType>(0))
      ? (max < std::numeric_limits<RealType>::infinity()
        ? get_absolute_gap() / max
        : max)
      : static_cast<RealType>(0);
  }


protected:
  // Problem specification
  const SizeType num_examples_;
  const SizeType num_tasks_;

  // Solver parameters
  const std::string solver_name_;
  SizeType check_gap_frequency_;
  SizeType max_num_epoch_;
  SizeType seed_;
  RealType epsilon_;

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


  virtual void BeginSolve();

  virtual void EndSolve();

  virtual void BeginEpoch();

  virtual bool EndEpoch();

  virtual void ComputeDualityGap();

  virtual void SolveExample(SizeType example) = 0;

  virtual void ComputePrimalDualObjectives() = 0;

};

}

#endif // SDCA_SOLVER_HPP

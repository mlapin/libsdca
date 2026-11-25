#ifndef SDCA_SOLVER_SOLVERDEF_H
#define SDCA_SOLVER_SOLVERDEF_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>

#include "sdca/core/project_base.h"
#include "sdca/utility/types.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

enum class solver_status {
  none = 0,
  solving,
  solved,
  no_progress,
  max_epoch,
  max_cpu_time,
  max_wall_time,
  failed
};

inline std::string solver_status_name(solver_status status_) {
  switch (status_) {
    case solver_status::none:
      return "none";
    case solver_status::solving:
      return "solving";
    case solver_status::solved:
      return "solved";
    case solver_status::no_progress:
      return "no_progress";
    case solver_status::max_epoch:
      return "max_epoch";
    case solver_status::max_cpu_time:
      return "max_cpu_time";
    case solver_status::max_wall_time:
      return "max_wall_time";
    case solver_status::failed:
      return "failed";
  }
  assert(false);
  return "unknown";
}

// How the epochs of the hinge objectives leave out work (stopping_criteria::
// shrinking); the entropy objectives always update every class of every
// example. heuristic: between full passes, skip the examples whose last
// step left their block unchanged and step a single-label block on its
// classes of nonzero dual. safe: after the evaluations that halve the
// certified gap, leave out for good the zero duals that the gap proves zero
// at the optimum (single-label hinges on features). both: the two. automatic:
// the default, both. Every evaluation covers every example and class.
enum class shrinking_mode { automatic = 0, off, heuristic, safe, both };

inline std::string shrinking_mode_name(const shrinking_mode mode) {
  switch (mode) {
    case shrinking_mode::automatic:
      return "auto";
    case shrinking_mode::off:
      return "off";
    case shrinking_mode::heuristic:
      return "heuristic";
    case shrinking_mode::safe:
      return "safe";
    case shrinking_mode::both:
      return "both";
  }
  assert(false);
  return "unknown";
}

struct stopping_criteria {
  // Evaluate every eval_epoch epochs (0: only at the end), and with tol > 0
  // up to twice in between where the gap's trend reaches tol: at most
  // max_epoch / eval_epoch + 2 evaluations, the start and the end included.
  size_type eval_epoch = 10;
  size_type max_epoch = 1000;
  // The relative duality gap of solver_status::solved (0: never solved).
  double tol = 1e-3;
  double max_cpu_time = 0;
  double max_wall_time = 0;
  bool eval_on_start = false;
  // Shrinking of the hinge objectives (shrinking_mode). The heuristic one
  // updates every class of every example in the first epoch of a solve and
  // in the epoch of each scheduled evaluation. Its epochs between full passes
  // are cheap, so a multiple of eval_epoch evaluates once the epochs since the
  // last evaluation took the steps of eval_epoch full passes, or once they
  // reach full_pass_epoch (with eval_epoch 0: a full pass every
  // full_pass_epoch epochs; 0 or 1: every epoch is a full pass).
  shrinking_mode shrinking = shrinking_mode::automatic;
  size_type full_pass_epoch = 50;
  // Recompute W = W0 + X A' at each evaluation (the dual is from A either way).
  bool recompute_primal = true;
  // The checks of a training kernel K, relative to sqrt(K_ii K_jj): |K_ij - K_ji| at most
  // kernel_symmetry, |K_ij| at most 1 + kernel_cauchy_schwarz (positive semidefinite K meet
  // both exactly). K comes from the caller, computed in any precision and order, so these are
  // defaults rather than roundings of the library: 1e-5 takes a float32 Gram of up to about
  // 80 features summed in any order (twice gamma_d(float) apart), 1e-3 rejects grossly
  // indefinite kernels only.
  double kernel_symmetry = 1e-5;
  double kernel_cauchy_schwarz = 1e-3;
  // Threads of the evaluations and input checks (the epochs are sequential):
  // the default takes SDCA_NUM_THREADS, else 0: automatic (the available
  // CPUs, as the work allows); 1 is serial.
  // Results are identical for every count.
  std::size_t num_threads = static_cast<std::size_t>(-1);
  // The seed of the order of the examples in the epochs; seeds that are equal
  // modulo 2^31 - 2 give the same order.
  std::uint64_t seed = 0;

  inline std::string to_string() const {
    std::ostringstream str;
    str << "tol: " << tol
        << ", "
           "eval_epoch: "
        << eval_epoch
        << ", "
           "max_epoch: "
        << max_epoch
        << ", "
           "max_cpu_time: "
        << max_cpu_time
        << ", "
           "max_wall_time: "
        << max_wall_time
        << ", "
           "eval_on_start: "
        << eval_on_start << ", seed: " << seed;
    return str.str();
  }
};

SDCA_ISA_END

#endif

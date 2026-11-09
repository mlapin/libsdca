#ifndef SDCA_SOLVER_SOLVERDEF_H
#define SDCA_SOLVER_SOLVERDEF_H

#include <cassert>
#include <cstddef>
#include <sstream>

#include "sdca/prox/proxdef.h"
#include "sdca/utility/types.h"

namespace sdca {

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
  // Evaluate every eval_epoch epochs (0: only at the end), and with epsilon > 0
  // up to twice in between where the gap's trend reaches epsilon: at most
  // max_epoch / eval_epoch + 2 evaluations, the start and the end included.
  size_type eval_epoch = 10;
  size_type max_epoch = 1000;
  double epsilon = 1e-3;
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
  // The hinge projections' precision (projection_precision); float64 by default.
  projection_precision projection = projection_precision::float64;
  // Threads of the evaluations and input checks (the epochs are sequential):
  // the default takes SDCA_NUM_THREADS, else 0: automatic (the available
  // CPUs, as the work allows); 1 is serial.
  // Results are identical for every count.
  std::size_t num_threads = static_cast<std::size_t>(-1);

  inline std::string to_string() const {
    std::ostringstream str;
    str << "epsilon: " << epsilon
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
        << eval_on_start;
    return str.str();
  }
};

}  // namespace sdca

#endif

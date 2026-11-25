#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include <atomic>

#include "sdca/math/blas.h"
#include "sdca/solver/solverdef.h"
#include "sdca/solver/eval/types.h"
#include "sdca/utility/logging.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace reporting {

template <typename Context>
inline void begin_solve(const Context& ctx) {
#if defined(SDCA_BLAS_FALLBACK)
  static std::atomic<bool> warned(false);
  if (logging::get_level() >= logging::level::warning && !warned.exchange(true)) {
    SDCA_LOG_WARNING << "libsdca found no BLAS and uses its built-in products; build with an "
                        "external BLAS (docs/API.md#blas) for faster predictions"
                     << std::endl;
  }
#endif
  SDCA_LOG_INFO << "Solve: " << ctx.to_string() << std::endl;
}

template <typename Context>
inline void end_solve(const Context& ctx) {
  if (ctx.status == solver_status::solved) {
    SDCA_LOG_INFO << "Solution: " << ctx.status_string() << std::endl;
  } else {
    if (logging::get_level() < logging::level::info) {
      SDCA_LOG_WARNING << "Solve: " << ctx.to_string() << std::endl;
    }
    SDCA_LOG_WARNING << "Solution: " << ctx.status_string() << std::endl;
  }
}

template <typename Context>
inline void end_epoch(const Context& ctx, const bool is_evaluated) {
  if (is_evaluated) return;
  SDCA_LOG_DEBUG << "  "
                    "epoch: "
                 << ctx.epoch
                 << ", "
                    "cpu_time: "
                 << ctx.cpu_time() << " (" << ctx.solve_time.cpu.elapsed << " + "
                 << ctx.eval_time.cpu.elapsed
                 << "), "
                    "wall_time: "
                 << ctx.wall_time() << " (" << ctx.solve_time.wall.elapsed << " + "
                 << ctx.eval_time.wall.elapsed << ")" << std::endl;
}

template <typename Result, typename Output>
inline void eval_created(const eval_train<Result, Output>& eval, const size_type) {
  SDCA_LOG_VERBOSE << "  " << eval.to_string() << std::endl;
}

template <typename Result, typename Output>
inline void eval_created(const eval_test<Result, Output>& eval, const size_type id) {
  SDCA_LOG_VERBOSE << "  eval on set #" << id << ": " << eval.to_string() << std::endl;
}

template <typename Result>
inline void solver_stop_failed(const Result absolute_gap, const Result eps_machine,
                               const Result eps_user) {
  SDCA_LOG_WARNING << "Warning: negative duality gap; "
                      "absolute_gap: "
                   << absolute_gap
                   << ", "
                      "eps_machine: "
                   << eps_machine
                   << ", "
                      "eps_user: "
                   << eps_user << std::endl;
}

inline void solver_stop_step_failed(const std::exception& e) {
  SDCA_LOG_WARNING << "Warning: coordinate step failed; " << e.what() << std::endl;
}

template <typename Result, typename Output>
inline void solver_stop_no_progress(const eval_train<Result, Output>& eval,
                                    const eval_train<Result, Output>& before) {
  SDCA_LOG_WARNING << "Warning: dual objective decreased; "
                      "dual: "
                   << eval.dual
                   << ", "
                      "dual_before: "
                   << before.dual
                   << ", "
                      "difference: "
                   << eval.dual - before.dual << std::endl;
}

}  // namespace reporting
SDCA_ISA_END

#endif

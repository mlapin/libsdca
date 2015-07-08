#ifndef SDCA_SOLVE_PRIMAL_SOLVER_H
#define SDCA_SOLVE_PRIMAL_SOLVER_H

#include "linalg/linalg.h"
#include "solver.h"

namespace sdca {

template <typename real_type, typename SolverHelperType>
class PrimalSolver : public solver<real_type> {

using solver<real_type>::num_examples_;
using solver<real_type>::num_tasks_;

public:
  PrimalSolver(
      const SolverHelperType &solver_helper,
      const std::size_t num_dimensions,
      const std::size_t num_examples,
      const std::size_t num_tasks,
      const real_type *features,
      const std::size_t *labels,
      real_type *primal_variables,
      real_type *dual_variables
    );

protected:
  SolverHelperType solver_helper_;
  const std::size_t num_dimensions_;
  const real_type *features_;
  const std::size_t *labels_;
  real_type *primal_variables_;
  real_type *dual_variables_;
  std::vector<real_type> norms_;
  std::vector<real_type> scores_;
  std::vector<real_type> dual_old_;
  const real_type diff_tolerance_;

  void BeginSolve() override;

  void SolveExample(std::size_t example) override;

  void ComputePrimalDualObjectives() override;

};

}

#endif

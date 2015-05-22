#ifndef PRIMAL_SOLVER_HPP
#define PRIMAL_SOLVER_HPP

#include <vector>

#include "dual_variables_helper.hpp"
#include "solver.hpp"

namespace sdca {

template <typename RealType, typename SolverHelperType>
class PrimalSolver : public Solver<RealType> {

using Solver<RealType>::num_examples_;
using Solver<RealType>::num_tasks_;

public:
  PrimalSolver(
      const SolverHelperType &solver_helper,
      const SizeType num_dimensions,
      const SizeType num_examples,
      const SizeType num_tasks,
      const RealType *features,
      const SizeType *labels,
      RealType *primal_variables,
      RealType *dual_variables
    );

protected:
  const SolverHelperType solver_helper_;
  const SizeType num_dimensions_;
  const RealType *features_;
  const SizeType *labels_;
  RealType *primal_variables_;
  RealType *dual_variables_;
  std::vector<RealType> norms_;
  std::vector<RealType> scores_;
  std::vector<RealType> dual_old_;
  const RealType diff_tolerance_;

  void BeginSolve() override;

  void SolveExample(SizeType example) override;

  void ComputePrimalDualObjectives() override;

};

}

#endif // PRIMAL_SOLVER_HPP

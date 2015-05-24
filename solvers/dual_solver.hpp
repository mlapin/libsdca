#ifndef DUAL_SOLVER_HPP
#define DUAL_SOLVER_HPP

#include <vector>

#include "dual_variables_helper.hpp"
#include "solver.hpp"

namespace sdca {

template <typename RealType, typename SolverHelperType>
class DualSolver : public Solver<RealType> {

using Solver<RealType>::num_examples_;
using Solver<RealType>::num_tasks_;

public:
  DualSolver(
      const SolverHelperType &solver_helper,
      const SizeType num_examples,
      const SizeType num_tasks,
      const RealType *gram_matrix,
      const SizeType *labels,
      RealType *dual_variables
    );

protected:
  SolverHelperType solver_helper_;
  const RealType *gram_matrix_;
  const SizeType *labels_;
  RealType *dual_variables_;
  std::vector<RealType> scores_;

  void BeginSolve() override;

  void SolveExample(SizeType example) override;

  void ComputePrimalDualObjectives() override;

};

}

#endif // DUAL_SOLVER_HPP

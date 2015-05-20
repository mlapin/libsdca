#ifndef DUAL_SOLVER_HPP
#define DUAL_SOLVER_HPP

#include <vector>

#include "dual_solver_helper.hpp"
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
    ) :
      Solver<RealType>::Solver(num_examples, num_tasks),
      solver_helper_(solver_helper),
      gram_matrix_(gram_matrix),
      labels_(labels),
      dual_variables_(dual_variables),
      scores_(num_tasks)
  {}

protected:
  const SolverHelperType solver_helper_;
  const RealType *gram_matrix_;
  const SizeType *labels_;
  RealType *dual_variables_;
  std::vector<RealType> scores_;

  void SolveExample(SizeType example) override;

  void ComputePrimalDualObjectives() override;

};

}

#endif // DUAL_SOLVER_HPP

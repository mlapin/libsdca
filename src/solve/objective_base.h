#ifndef SDCA_SOLVE_OBJECTIVE_BASE_H
#define SDCA_SOLVE_OBJECTIVE_BASE_H

#include "util/util.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Summation>
struct objective_base {
  typedef Data data_type;
  typedef Result result_type;
  typedef Summation summation_type;

  const Result p_loss_coeff;
  const Summation sum;

  objective_base(
      const Result __coeff,
      const Summation __sum
    ) :
      p_loss_coeff(__coeff),
      sum(__sum)
  {}

  inline std::string precision_string() const {
    std::ostringstream str;
    str << "precision = " << type_traits<Result>::name() << ", "
      "data_precision = " << type_traits<Data>::name() << ", "
      "summation = " << sum.name();
    return str.str();
  }

  inline Result
  regularizer(
      const blas_int num_tasks,
      const Data* variables,
      const Data* scores
    ) const {
    return static_cast<Result>(sdca_blas_dot(num_tasks, scores, variables));
  }

  inline Result
  dual_loss(
      const blas_int,
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0]);
  }

  inline void
  update_loss(
      Result& p_loss
    ) const {
    p_loss *= p_loss_coeff;
  }

  inline void
  update_all(
      Result& p_loss,
      Result& d_loss,
      Result& regul,
      Result& p_objective,
      Result& d_objective,
      Result& duality_gap
    ) const {
    p_loss *= p_loss_coeff;
    duality_gap = p_loss - d_loss + regul;
    regul *= static_cast<Result>(0.5);
    p_objective = p_loss + regul;
    d_objective = d_loss - regul;
  }
};

}

#endif

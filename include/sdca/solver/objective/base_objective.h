#ifndef SDCA_SOLVER_OBJECTIVE_BASE_OBJECTIVE_H
#define SDCA_SOLVER_OBJECTIVE_BASE_OBJECTIVE_H

#include <sstream>

#include "sdca/math/blas.h"
#include "sdca/utility/type_name.h"

namespace sdca {

template <typename Data,
          typename Result>
struct base_objective {
  typedef Data data_type;
  typedef Result result_type;

  const Result p_loss_coeff;


  explicit base_objective(const Result __p_loss_coeff)
    : p_loss_coeff(__p_loss_coeff)
  {}


  inline std::string precision_string() const {
    std::ostringstream str;
    str << "precision ("
        << type_name<Data>() << ", "
        << type_name<Result>() << ")";
    return str.str();
  }


  inline Result
  regularizer_primal(
      const blas_int num_dim,
      const Data* variables
    ) const {
    return static_cast<Result>(sdca_blas_dot(num_dim, variables, variables));
  }


  inline Result
  regularizer_dual(
      const blas_int num_classes,
      const Data* variables,
      const Data* scores
    ) const {
    return static_cast<Result>(sdca_blas_dot(num_classes, variables, scores));
  }


  inline Result
  dual_loss(
      const blas_int,
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0]);
  }


  inline void
  update_primal_loss(
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
    duality_gap = p_loss + regul - d_loss;
    regul *= static_cast<Result>(0.5);
    p_objective = p_loss + regul;
    d_objective = d_loss - regul;
  }

};

}

#endif

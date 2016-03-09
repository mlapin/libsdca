#ifndef SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H
#define SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H

#include <sstream>

#include "sdca/math/blas.h"
#include "sdca/utility/type_name.h"

namespace sdca {

template <typename Data,
          typename Result>
struct objective_base {
  typedef Data data_type;
  typedef Result result_type;

  const Result coeff_primal_loss;


  explicit objective_base(const Result __coeff_primal_loss = 1)
    : coeff_primal_loss(__coeff_primal_loss)
  {}


  inline std::string precision_string() const {
    std::ostringstream str;
    str << "precision ("
        << type_name<Data>() << ", "
        << type_name<Result>() << ")";
    return str.str();
  }


  template <typename Int>
  inline Result
  regularizer_primal(
      const Int num_dimensions,
      const Data* variables
    ) const {
    return static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_dimensions), variables, variables));
  }


  template <typename Int>
  inline Result
  regularizer_dual(
      const Int num_classes,
      const Data* variables,
      const Data* scores
    ) const {
    return static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_classes), variables, scores));
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int,
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0]);
  }


  inline void
  update_primal_loss(
      Result& primal_loss
    ) const {
    primal_loss *= coeff_primal_loss;
  }


  inline void
  update_all(
      Result& primal,
      Result& dual,
      Result& primal_loss,
      Result& dual_loss,
      Result& regularizer
    ) const {
    primal_loss *= coeff_primal_loss;
    regularizer *= static_cast<Result>(0.5);
    primal = primal_loss + regularizer;
    dual = dual_loss - regularizer;
  }

};

}

#endif

#ifndef SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H
#define SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H

#include <sstream>

#include "sdca/math/blas.h"
#include "sdca/utility/types.h"

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


  inline std::string to_string() const {
    std::ostringstream str;
    str << "objective (precision: " << type_name<Result>() << ")";
    return str.str();
  }


  template <typename Int>
  void update_dual_variables(
      const Int,  // num_classes
      const Data, // norm2
      Data*,      // variables
      Data*       // scores
      ) const {}


  template <typename Int>
  inline Result
  primal_loss(
      const Int,  // num_classes
      Data*       // scores
    ) const {}


  template <typename Int>
  inline Result
  dual_loss(
      const Int,  // num_classes
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0]);
  }


  template <typename Int>
  inline void
  regularizers_primal(
      const Int num_dimensions,
      const Data* variables,
      Result& primal_regularizer,
      Result& dual_regularizer
    ) const {
    auto regul = static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_dimensions), variables, variables)) / 2;
    primal_regularizer += regul;
    dual_regularizer += regul;
  }


  template <typename Int>
  inline void
  regularizers_primal(
      const Int num_dimensions,
      const Data* variables,
      const Data* initial_variables,
      Result& primal_regularizer,
      Result& dual_regularizer
    ) const {
    auto regul = static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_dimensions), variables, variables)) / 2;
    auto prox =  static_cast<Result>(sdca_blas_dot(
        static_cast<blas_int>(num_dimensions), initial_variables, variables));
    primal_regularizer += regul - prox;
    dual_regularizer += regul;
  }


  template <typename Int>
  inline void
  regularizers_dual(
      const Int num_classes,
      const Data* variables,
      const Data* scores,
      Result& primal_regularizer,
      Result& dual_regularizer
    ) const {
    auto regul = static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_classes), variables, scores)) / 2;
    primal_regularizer += regul;
    dual_regularizer += regul;
  }


  inline void
  update_primal_loss(
      Result& __primal_loss
    ) const {
    __primal_loss *= coeff_primal_loss;
  }


  inline void
  update_all(
      Result& __primal,
      Result& __dual,
      Result& __primal_loss,
      Result& __dual_loss,
      Result& __primal_regularizer,
      Result& __dual_regularizer
    ) const {
    __primal_loss *= coeff_primal_loss;
    __primal = __primal_loss + __primal_regularizer;
    __dual = __dual_loss - __dual_regularizer;
  }

};

}

#endif

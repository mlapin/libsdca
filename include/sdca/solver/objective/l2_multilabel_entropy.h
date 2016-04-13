#ifndef SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_ENTROPY_H
#define SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_ENTROPY_H

#include "sdca/math/log_exp.h"
#include "sdca/prox/two_entropy.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

template <typename Data,
          typename Result>
struct l2_multilabel_entropy
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;

  const Result c_log_c;


  l2_multilabel_entropy(
      const Result __c
    ) :
      base::objective_base(__c),
      c(__c),
      c_log_c(x_log_x(__c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_entropy (c: " << c <<
           ", precision: " << type_name<Result>() << ")";
    return str.str();
  }


  template <typename Int>
  void update_dual_variables(
      const Int num_classes,
      const Int num_labels,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *pos_first(variables), *pos_last(variables + num_labels);
    Data *neg_first(variables + num_labels), *neg_last(variables + num_classes);
    Data *pos_scores(scores), *neg_scores(scores + num_labels);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), 1, scores, -norm2, variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    Result alpha(c * static_cast<Result>(norm2));
    prox_two_entropy(pos_first, pos_last, neg_first, neg_last,
                     pos_scores, neg_scores, alpha);

    // 3. Recover the updated variables
    Data a(static_cast<Data>(-c));
    Data b(static_cast<Data>(c / static_cast<Result>(num_labels)));
    std::for_each(pos_first, pos_last, [=](Data &x){ x = a * x + b; });
    sdca_blas_scal(static_cast<blas_int>(num_classes - num_labels),
                   a, neg_first);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      const Int num_labels,
      Data* scores
    ) const {
    Data *pos_first(scores), *pos_last(scores + num_labels);
    Data *neg_last(scores + num_classes);

    Result lse = log_sum_exp<Result>(pos_first, neg_last);
    Result avg = std::accumulate(pos_first, pos_last, static_cast<Result>(0))
               / static_cast<Result>(num_labels);
    return lse - avg;
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int num_classes,
      const Int num_labels,
      const Data* variables
    ) const {
    Result d_loss(0), p(static_cast<Result>(num_labels));

    std::for_each(variables, variables + num_labels,
      [&](const Result a){ d_loss -= x_log_x(c - p * a); });
    d_loss /= p;

    std::for_each(variables + num_labels, variables + num_classes,
      [&](const Result a){ d_loss -= x_log_x(-a); });

    Result sum = std::accumulate(variables, variables + num_labels,
                                 static_cast<Result>(0));
    d_loss += c_log_c + std::log(p) * (c - sum);
    return d_loss;
  }

};

}

#endif

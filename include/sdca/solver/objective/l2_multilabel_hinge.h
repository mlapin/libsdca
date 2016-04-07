#ifndef SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_HINGE_H
#define SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_HINGE_H

#include "sdca/types.h"
#include "sdca/prox/two_simplex.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

template <typename Data,
          typename Result>
struct l2_multilabel_hinge
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;


  l2_multilabel_hinge(
      const Result __c
    ) :
      base::objective_base(__c),
      c(__c)
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_hinge (c: " << c <<
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
    Data a(1 / norm2);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), a, scores, -1, variables);
    a /= 2;
    std::for_each(pos_first, pos_last, [=](Data &x){ x = a - x; });
    std::for_each(neg_first, neg_last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_two_simplex(pos_first, pos_last, neg_first, neg_last,
                     pos_scores, neg_scores, c);

    // 3. Recover the updated variables
    sdca_blas_scal(static_cast<blas_int>(num_classes - num_labels),
                   static_cast<Data>(-1), neg_first);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      const Int num_labels,
      Data* scores
    ) const {
    Data *pos_first(scores), *pos_last(scores + num_labels);
    Data *neg_first(scores + num_labels), *neg_last(scores + num_classes);

    Data min_pos = *std::min_element(pos_first, pos_last);
    Data max_neg = *std::max_element(neg_first, neg_last);

    Result zero(0), loss = static_cast<Result>(max_neg - min_pos + 1);
    return std::max(zero, loss);
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int,
      const Int num_labels,
      const Data* variables
    ) const {
    return std::accumulate(variables, variables + num_labels,
                           static_cast<Result>(0));
  }

};


template <typename Data,
          typename Result>
struct l2_multilabel_hinge_smooth
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const Result gamma;

  const Result gamma_div_c;
  const Result gamma_div_2c;


  l2_multilabel_hinge_smooth(
      const Result __c,
      const Result __gamma
    ) :
      base::objective_base(__c / __gamma),
      c(__c),
      gamma(__gamma),
      gamma_div_c(__gamma / __c),
      gamma_div_2c(__gamma / (2 * __c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_hinge (c: " << c << ", gamma: " << gamma <<
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
    Data a = static_cast<Data>(1 / (static_cast<Result>(norm2) + gamma_div_c));
    Data b = static_cast<Data>(static_cast<Result>(norm2)
                                 / (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), a, scores, -b, variables);
    a /= 2;
    std::for_each(pos_first, pos_last, [=](Data &x){ x = a - x; });
    std::for_each(neg_first, neg_last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_two_simplex(pos_first, pos_last, neg_first, neg_last,
                     pos_scores, neg_scores, c);

    // 3. Recover the updated variables
    sdca_blas_scal(static_cast<blas_int>(num_classes - num_labels),
                   static_cast<Data>(-1), neg_first);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      const Int num_labels,
      Data* scores
    ) const {
    Data *pos_first(scores), *pos_last(scores + num_labels);
    Data *neg_first(scores + num_labels), *neg_last(scores + num_classes);

    const Data a(0.5);
    std::for_each(pos_first, pos_last, [=](Data &x){ x = a - x; });
    std::for_each(neg_first, neg_last, [=](Data &x){ x += a; });

    auto t = thresholds_two_simplex(pos_first, pos_last, neg_first, neg_last,
                                    gamma);

    Result pos_hp = dot_x_prox(t.first, pos_first, pos_last);
    Result pos_pp = dot_prox_prox(t.first, pos_first, pos_last);
    Result neg_hp = dot_x_prox(t.second, neg_first, neg_last);
    Result neg_pp = dot_prox_prox(t.second, neg_first, neg_last);

    return (pos_hp + neg_hp) - static_cast<Result>(0.5) * (pos_pp + neg_pp);
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int num_classes,
      const Int num_labels,
      const Data* variables
    ) const {
    Result loss = std::accumulate(variables, variables + num_labels,
                                  static_cast<Result>(0));
    Result smoothing = gamma_div_2c * static_cast<Result>(sdca_blas_dot(
      static_cast<blas_int>(num_classes), variables, variables));
    return loss - smoothing;
  }

};

}

#endif

#ifndef SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_NN_FEATURES_H
#define SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_NN_FEATURES_H

#include "sdca/math/log_exp.h"
#include "sdca/prox/two_entropy.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

/*
 * Learn non-negative features optimizing the l2 regularized softmax loss.
 */
template <typename Data,
          typename Result>
struct l2_entropy_nn_features
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;

  const Result c_log_c;


  l2_entropy_nn_features(
      const Result __c
    ) :
      base::objective_base(__c),
      c(__c),
      c_log_c(x_log_x(__c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_entropy_nn_features (c: " << c <<
           ", precision: " << type_name<Result>() << ")";
    return str.str();
  }


  /*
   * variables <-
   *    argmin_a C L^*(Y_i, -1/C a) + lambda/2 ||a - variables||^2
   */
  template <typename Int>
  void prox_f(
      const Int num_classes,
      const Int num_labels,
      const Data lambda,
      Data* variables,
      Data* scores
      ) const {
    Data *pos_first(variables), *pos_last(variables + num_labels);
    Data *neg_first(variables + num_labels), *neg_last(variables + num_classes);
    Data *pos_scores(scores), *neg_scores(scores + num_labels);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_scal(static_cast<blas_int>(num_classes), -lambda, variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    Result alpha(c * static_cast<Result>(lambda));
    prox_two_entropy(pos_first, pos_last, neg_first, neg_last,
                     pos_scores, neg_scores, alpha);

    // 3. Recover the updated variables
    Data a(static_cast<Data>(-c));
    Data b(static_cast<Data>(c / static_cast<Result>(num_labels)));
    std::for_each(pos_first, pos_last, [=](Data &x){ x = a * x + b; });
    sdca_blas_scal(static_cast<blas_int>(num_classes - num_labels),
                   a, neg_first);
  }


  /*
   * This is also the gradient of the regularizer, since
   *    X = grad(g(W * A)),
   * where X are the features (primal variables),
   * g(U) = 1/2 ||max{0, X0 + U}||^2 is the regularizer,
   * W is the model weights matrix,
   * A is the matrix of dual variables.
   */
  template <typename Int>
  void compute_features(
      const Int num_dimensions,
      const Int num_classes,
      const Data* W,
      const Data* x_i_0,
      const Data* variables,
      Data* x_i
      ) const {
    const blas_int D = static_cast<blas_int>(num_dimensions);
    const blas_int M = static_cast<blas_int>(num_classes);

    // x_i = x_i^0 + Wa
    sdca_blas_copy(D, x_i_0, x_i);
    sdca_blas_gemv(D, M, W, variables, x_i, CblasNoTrans, 1, 1);

    // x_i = max{0, x_i^0 + Wa}
    std::for_each(x_i, x_i + num_dimensions,
                  [](Data &x){ x = std::max<Data>(0, x); });
  }


  template <typename Int>
  void compute_features(
      const Int num_dimensions,
      const Int num_examples,
      const Int num_classes,
      const Data* W,
      const Data* A,
      const Data* X0,
      Data* X
      ) const {
    // Let X = max{0, X0 + W * A}
    auto D = static_cast<blas_int>(num_dimensions);
    auto N = static_cast<blas_int>(num_examples);
    auto M = static_cast<blas_int>(num_classes);
    sdca_blas_copy(D * N, X0, X);
    sdca_blas_gemm(D, N, M, W, D, A, M, X, CblasNoTrans, CblasNoTrans, 1, 1);
    std::for_each(X, X + num_dimensions * num_examples,
                  [](Data &x){ x = std::max<Data>(0, x); });
  }


  /*
   * variables <-
   *    argmin_a g(a) + lambda/2 ||a - variables||^2,
   * where
   *    g(a) = 1/2 ||max{0, x_i^0 + Wa}||^2
   */
  template <typename Int>
  void prox_g(
      const Int num_dimensions,
      const Int num_classes,
      const Data lambda,
      const Data* W,
      const Data* x_i_0,
      Data* variables,
      Data* x_i
      ) const {
    const blas_int D = static_cast<blas_int>(num_dimensions);
    const blas_int M = static_cast<blas_int>(num_classes);

    compute_features(num_dimensions, num_classes, W, x_i_0, variables, x_i);

    // Apply shrinkage
    // (note that lambda is inverted here, therefore not lambda / (1 + lambda))
    Data coeff = - 1 / (1 + lambda);
    sdca_blas_gemv(D, M, W, x_i, variables, CblasTrans, coeff, 1);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      Data* scores
    ) const {
    const Int num_labels = 1;
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
      const Data* variables
    ) const {
    const Int num_labels = 1;
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

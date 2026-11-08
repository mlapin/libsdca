#ifndef SDCA_SOLVER_CONTEXT_H
#define SDCA_SOLVER_CONTEXT_H

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include "sdca/solver/data.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/solverdef.h"
#include "sdca/utility/stopwatch.h"

namespace sdca {

// k of a top-k objective must suit the classes of the training set.
template <typename Objective>
inline void check_k(const Objective&, size_type,
                    typename std::enable_if<!has_param_k<Objective>::value>::type* = 0) {}

template <typename Objective>
inline void check_k(const Objective& obj, const size_type num_classes,
                    typename std::enable_if<has_param_k<Objective>::value>::type* = 0) {
  detail::check_topk(obj.k, num_classes);
}

namespace detail {

// The entries of W in a training set: dim-by-m on features, none on a kernel.
template <typename Data, typename Output, typename Evaluation>
inline size_type primal_size(const dataset<feature_input<Data>, Output, Evaluation>& d) {
  return d.num_dimensions() * d.num_classes();
}
template <typename Data, typename Output, typename Evaluation>
inline size_type primal_size(const dataset<kernel_input<Data>, Output, Evaluation>&) {
  return 0;
}

// The iterate over a caller's buffer: the buffer itself in its own type, else
// a copy in T (shared by copies of the context, as the buffer is).
template <typename T>
inline T* wide_iterate(T* buffer, size_type, std::shared_ptr<std::vector<T>>&) {
  return buffer;
}
template <typename T, typename S>
inline T* wide_iterate(S* buffer, const size_type size, std::shared_ptr<std::vector<T>>& wide) {
  if (buffer == nullptr) return nullptr;
  wide = std::make_shared<std::vector<T>>(buffer, buffer + size);
  return wide->data();
}

// Copies of a wide iterate from and to the caller's buffer (rounded once);
// nothing where the two are one.
template <typename T>
inline void load_iterate(const T*, T*, size_type) {}
template <typename T, typename S>
inline void load_iterate(const S* buffer, T* wide, const size_type size) {
  if (buffer) std::copy(buffer, buffer + size, wide);
}
template <typename T>
inline void store_iterate(const T*, T*, size_type) {}
template <typename T, typename S>
inline void store_iterate(const T* wide, S* buffer, const size_type size) {
  if (buffer)
    for (size_type j = 0; j < size; ++j) buffer[j] = static_cast<S>(wide[j]);
}

}  // namespace detail

// What the evaluations and the screening read of a context: the caller's
// buffers (the stored iterate, as returned) in place of the iterate.
template <typename Context>
struct stored_iterate {
  typedef typename Context::data_type data_type;
  typedef data_type variable_type;

  const typename Context::train_set_type& train;
  const typename Context::objective_type& objective;
  stopping_criteria criteria;
  const data_type* primal_initial;
  data_type* primal_variables;
  data_type* dual_variables;
  const size_type epoch;
  const stopwatch& solve_time;
  const stopwatch& eval_time;
};

/*
 * The iterate of a fit, W (dim-by-m, features only) and the duals (m-by-n),
 * is in variable_type: the wider of the data and result types. The caller's buffers (primal_buffer, dual_buffer) hold data_type: with
 * float data and double results the iterate is a double copy, which the
 * solver loads from the buffers at the start of a solve. Each evaluation
 * stores it, rounded once, to the buffers and evaluates those, the values
 * returned; the iterate itself keeps its precision. In float, a step's change
 * of W or of a dual is often below their rounding near the optimum and lost.
 */
template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
struct solver_context {
  typedef Data data_type;
  typedef Result result_type;
  typedef Input<Data> input_type;
  typedef Output output_type;
  typedef Objective<Data, Result> objective_type;

  typedef typename std::common_type<Data, Result>::type variable_type;
  // The objective of the steps, in variable_type.
  typedef Objective<variable_type, Result> step_objective_type;

  typedef eval_type<Result> eval_result_type;
  typedef eval_train<eval_result_type, Output> train_eval_type;
  typedef eval_test<eval_result_type, Output> test_eval_type;

  typedef dataset<input_type, output_type, train_eval_type> train_set_type;
  typedef dataset<input_type, output_type, test_eval_type> test_set_type;

  train_set_type train;
  std::vector<test_set_type> test;

  objective_type objective;
  step_objective_type step_objective;
  stopping_criteria criteria;

  const data_type* primal_initial = nullptr;
  data_type* primal_buffer = nullptr;
  data_type* dual_buffer = nullptr;
  // The wide copies of the iterate, if any (before the pointers into them).
  std::shared_ptr<std::vector<variable_type>> primal_wide_;
  std::shared_ptr<std::vector<variable_type>> dual_wide_;
  variable_type* primal_variables = nullptr;
  variable_type* dual_variables = nullptr;

  solver_status status = solver_status::none;
  size_type epoch = 0;
  stopwatch solve_time;
  stopwatch eval_time;

  solver_context(train_set_type&& train_set_, objective_type&& objective_,
                 data_type* dual_variables_, data_type* primal_variables_ = nullptr,
                 const data_type* primal_initial_ = nullptr)
      : train(std::move(train_set_)),
        objective(std::move(objective_)),
        step_objective(objective),
        primal_initial(primal_initial_),
        primal_buffer(primal_variables_),
        dual_buffer(dual_variables_),
        primal_variables(detail::wide_iterate(primal_buffer, primal_size(), primal_wide_)),
        dual_variables(detail::wide_iterate(dual_buffer, dual_size(), dual_wide_)) {
    check_k(objective, train.num_classes());
  }

  size_type primal_size() const { return primal_buffer ? detail::primal_size(train) : 0; }
  size_type dual_size() const { return train.num_examples() * train.num_classes(); }

  // The iterate from the caller's buffers, and back to them (see above).
  void load_variables() {
    detail::load_iterate(primal_buffer, primal_variables, primal_size());
    detail::load_iterate(dual_buffer, dual_variables, dual_size());
  }
  void store_variables() const {
    detail::store_iterate(primal_variables, primal_buffer, primal_size());
    detail::store_iterate(dual_variables, dual_buffer, dual_size());
  }
  // The stored iterate (see store_variables) for the evaluations. A wide W
  // is stored as it is, not rebuilt from the stored duals (recompute_primal):
  // their rounding would enter W in every sum, and the wide W drifts only by
  // the rounding of its own type.
  stored_iterate<solver_context> stored() const {
    stopping_criteria c = criteria;
    c.recompute_primal = c.recompute_primal && std::is_same<variable_type, data_type>::value;
    return {train,       objective, c,          primal_initial, primal_buffer,
            dual_buffer, epoch,     solve_time, eval_time};
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << objective.to_string() << ", " << "stopping_criteria (" << criteria.to_string() << "), "
        << train.to_string()
        << ", "
           "is_prox: "
        << is_prox();
    return str.str();
  }

  inline std::string status_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "status: " << solver_status_name(status);
    if (train.evals.size() > 0) {
      str << ", " << train.evals.back().to_string(true);
    }
    str << ", cpu_time: " << cpu_time() << " (solve: " << solve_time.cpu.elapsed
        << ", eval: " << eval_time.cpu.elapsed
        << ")"
           ", wall_time: "
        << wall_time() << " (solve: " << solve_time.wall.elapsed
        << ", eval: " << eval_time.wall.elapsed << ")";
    return str.str();
  }

  void add_test(input_type&& in, output_type&& out) {
    validate_test_input(train.in, in);
    if (out.num_classes != train.num_classes())
      throw std::invalid_argument("Training and test class counts differ");
    test.emplace_back(std::move(in), std::move(out));
  }

  std::string status_name() const { return solver_status_name(status); }

  bool is_dual() const { return primal_variables == nullptr; }

  bool is_prox() const { return primal_initial != nullptr; }

  double cpu_time() const { return solve_time.cpu.elapsed + eval_time.cpu.elapsed; }

  double wall_time() const { return solve_time.wall.elapsed + eval_time.wall.elapsed; }

  double cpu_time_now() const { return solve_time.cpu.elapsed_now() + eval_time.cpu.elapsed_now(); }

  double wall_time_now() const {
    return solve_time.wall.elapsed_now() + eval_time.wall.elapsed_now();
  }
};

template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline solver_context<Data, Result, Input, Output, Objective> make_context(
    Input<Data>&& in, Output&& out, Objective<Data, Result>&& objective, Data* dual_variables,
    Data* primal_variables) {
  return solver_context<Data, Result, Input, Output, Objective>(
      make_dataset_train<eval_type<Result>>(std::move(in), std::move(out)), std::move(objective),
      dual_variables, primal_variables);
}

template <typename Data, typename Result, typename Output,
          template <typename, typename> class Objective>
inline solver_context<Data, Result, kernel_input, Output, Objective> make_context(
    kernel_input<Data>&& in, Output&& out, Objective<Data, Result>&& objective,
    Data* dual_variables) {
  return solver_context<Data, Result, kernel_input, Output, Objective>(
      make_dataset_train<eval_type<Result>>(std::move(in), std::move(out)), std::move(objective),
      dual_variables);
}

}  // namespace sdca

#endif

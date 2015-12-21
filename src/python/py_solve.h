#ifndef SDCA_PYTHON_PY_SOLVE_H
#define SDCA_PYTHON_PY_SOLVE_H

#include "py_util.h"
#include "../solve/solve.h"

using namespace sdca;

namespace py_solve {

template <typename Result>
inline void
add_records(
    const std::vector<train_point<Result>>& records,
    modelInfo<Result>* info,
    solveOpts<Result>* opts
  ) {
  if (opts->return_records) {
    info->num_records = records.size();
    const size_t num_records = info->num_records;
    info->records = new Result[num_records * 13];
    size_type i = 0;
    for (auto& a : records) {
      info->records[13*i + 0] = a.epoch;
      info->records[13*i + 1] = a.primal;
      info->records[13*i + 2] = a.dual;
      info->records[13*i + 3] = a.gap;
      info->records[13*i + 4] = a.primal_loss;
      info->records[13*i + 5] = a.dual_loss;
      info->records[13*i + 6] = a.regularizer;
      info->records[13*i + 7] = a.wall_time;
      info->records[13*i + 8] = a.cpu_time;
      info->records[13*i + 9] = a.solve_wall_time;
      info->records[13*i + 10] = a.solve_cpu_time;
      info->records[13*i + 11] = a.eval_wall_time;
      info->records[13*i + 12] = a.eval_cpu_time;
      ++i;
    }
  }
}

template <typename Result>
inline void
add_evaluations(
    const std::vector<std::vector<test_point<Result>>>& evals,
    modelInfo<Result>* info,
    solveOpts<Result>* opts
  ) {
  if (opts->return_evals) {
    info->num_dataset_evals = evals.size();
    info->num_evals = evals[0].size();
    info->evals = new Result[info->num_dataset_evals * info->num_evals * (1 + info->num_classes)];
    size_t i = 0;
    for (auto& dataset_evals : evals) {
      for (auto& a : dataset_evals) {
        info->evals[i] = a.loss;
        ++i;
        for (auto& acc : a.accuracy){
          info->evals[i] = acc;
          ++i;
        }
      }
    }
  }
}

template <typename Solver,
          typename Result>
inline void
solve_objective_add_info(
    Solver solver,
    modelInfo<Result>* info,
    solveOpts<Result>* opts
  ) {
  solver.solve();
  info->status = solver.status_name().c_str();
  info->primal = solver.primal();
  info->dual = solver.dual();
  info->absolute_gap = solver.absolute_gap();
  info->relative_gap = solver.relative_gap();
  info->epoch = solver.epoch();
  info->wall_time = solver.wall_time();
  info->cpu_time = solver.cpu_time();
  info->solve_wall_time = solver.solve_wall_time();
  info->solve_cpu_time = solver.solve_cpu_time();
  info->eval_wall_time = solver.eval_wall_time();
  info->eval_cpu_time = solver.eval_cpu_time();
  add_records(solver.records(), info, opts);
  add_evaluations(solver.evaluations(), info, opts);
}

template <typename Objective,
          typename Data,
          typename Result>
inline void
make_solver_solve(
    solver_context<Data>& context,
    modelInfo<Result>* info,
    solveOpts<Result>* opts,
    const Objective& objective
  ) {
  if (context.is_dual) {
    solve_objective_add_info(
        dual_solver<Objective, Data, Result>(objective, context), info, opts);
  } else {
    solve_objective_add_info(
        primal_solver<Objective, Data, Result>(objective, context), info, opts);
  }
}

template <typename Data,
          typename Result>
inline void
set_variables(
    solver_context<Data>& context,
    modelInfo<Result>* info
  ) {
  context.dual_variables = static_cast<Data*>(info->A);
  if (!context.is_dual) {
    context.primal_variables = static_cast<Data*>(info->W);
  }
}

template <typename Data,
          typename Result,
          typename Summation>
void
py_main(
    std::vector<dataset<Data>>* datasets,
    modelInfo<Result>* info,
    solveOpts<Result>* opts,
    Summation& sum
    ) {
  solver_context<Data> context;
  context.is_dual = opts->is_dual;
  set_datasets(datasets, context);
  set_stopping_criteria(opts, context);
  auto trn_data = context.datasets.front();
  set_variables(context, info);

  if (!context.is_dual) {
    info->num_dimensions = trn_data.num_dimensions;
  }
  info->num_examples = trn_data.num_examples;
  info->num_classes = trn_data.num_classes;
  info->objective = opts->objective;
  info->C = opts->C;
  if (info->objective == "msvm" ||
      info->objective == "multi_svm") {
    make_solver_solve(context, info, opts,
      l2_topk_hinge<Data, Result, Summation>(opts->k, opts->C, sum));
  } else if (info->objective == "l2_hinge_topk" ||
             info->objective == "topk_hinge_alpha" ||
             info->objective == "topk_svm") {
    info->k = opts->k;
    make_solver_solve(context, info, opts,
      l2_hinge_topk<Data, Result, Summation>(opts->k, opts->C, sum));
  } else if (info->objective == "l2_topk_hinge" ||
             info->objective == "topk_hinge_beta") {
    info->k = opts->k;
    make_solver_solve(context, info, opts,
      l2_topk_hinge<Data, Result, Summation>(opts->k, opts->C, sum));
  } else {
    throw std::runtime_error(
        "Invalid value passed to py_solve in opts.objective."
        "Valid options are: msvm, multi_svm, l2_hinge_topk, "
        "topk_hinge_alpha, topk_svm, l2_topk_hinge, or "
        "topk_hinge_beta."
        );
  }
  info->check_on_start = context.criteria.check_on_start;
  info->check_epoch = context.criteria.check_epoch;
  info->max_epoch = context.criteria.max_epoch;
  info->max_cpu_time = context.criteria.max_cpu_time;
  info->max_wall_time = context.criteria.max_wall_time;
  info->epsilon = context.criteria.epsilon;
  info->log_level = logging::get_level_name();
  info->log_format = logging::get_format_name();
  info->summation = sum.name();
  info->precision = type_traits<Result>::name();
}

template <typename Data,
          typename Result>
inline void
py_main(
    std::vector<dataset<Data>>* datasets,
    modelInfo<Result>* info,
    solveOpts<Result>* opts
    ) {
  if (opts->summation == "standard" || opts->summation == "default") {
    std_sum<Data*, Result> sum;
    py_main<Data, Result, std_sum<Data*, Result>>(datasets, info, opts, sum);
  } else if (opts->summation == "kahan") {
    kahan_sum<Data*, Result> sum;
    py_main<Data, Result, kahan_sum<Data*, Result>>(datasets, info, opts, sum);
  } else {
    throw std::runtime_error(
        "Invalid value passed to py_solve in opts.summation."
        "Valid options are: standard, default, or kahan."
        );
  }
}

// currently only allow double precision input matrices
template <typename Result>
void
py_solve_(
    std::vector<dataset<double>>* datasets,
    modelInfo<Result>* info,
    solveOpts<Result>* opts
    ) {
  logging::format_push();
  py_main<double, double>(datasets, info, opts);
  logging::format_pop();
}

}

#endif

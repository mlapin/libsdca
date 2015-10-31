#include <iostream>

#include "mex_util.h"
#include "solve/solvedef.h"
#include "util/util.h"

#ifndef MEX_GD
#define MEX_GD "mex_gd"
#endif
#ifndef LIBSDCA_VERSION
#define LIBSDCA_VERSION "0.0.0"
#endif

using namespace sdca;

inline void
printUsage() {
  mexPrintf("Usage: model = %s(X,Y,opts);\n", MEX_GD);
}

template <typename Data>
double
grad_topk_softmax_nonconvex(
    const size_type num_dimensions,
    const size_type num_examples,
    const size_type num_tasks,
    const Data* X,
    const std::vector<size_type>& Y,
    const double C,
    const size_type k,
    const Data* W,
    Data* grad,
    std::vector<Data>& scores,
    std::vector<Data>& scratch,
    std::vector<size_type>& idx
    ) {
  auto ifirst = idx.begin() + 1; // ground truth at 0, start at 1
  auto ikth = idx.begin() + k; // first + k - 1
  auto ilast = idx.end();
  auto kth = scores.begin() + k;
  auto last = scores.end();
  const blas_int D = static_cast<blas_int>(num_dimensions);
  const blas_int T = static_cast<blas_int>(num_tasks);

  double objective(0);
  sdca_blas_copy(D * T, W, grad);
  for (size_type i = 0; i < num_examples; ++i) {
    const Data* x_i = X + num_dimensions * i;
    const size_type label = Y[i];
    std::iota(idx.begin(), idx.end(), 0);

    // Compute scores
    sdca_blas_gemv(D, T, W, x_i, &scratch[0], CblasTrans);
    std::swap(idx[0], idx[label]);

    // Find k largest elements (re-order indexes)
    std::nth_element(ifirst, ikth, ilast,
      [&scratch](size_type i1, size_type i2)
      { return scratch[i1] > scratch[i2]; });

    // Compute the re-ordered scores, starting with the kth largest one
    scores[0] = scratch[label];
    for (size_type ix = k; ix < num_tasks; ++ix) {
      scores[ix] = scratch[idx[ix]];
    }

    // Pre-compute exp(score_j - M), where M is the kth largest score
    Data M(*kth); // kth is the maximum in [kth, last)
    std::for_each(kth + 1, last, [=](Data &x){ x = std::exp(x - M); });

    // Compute the log(1 + sum exp) loss and the intermediate terms
    double s = std::accumulate(kth + 1, last, 0.0);
    double a = static_cast<double>(M) - static_cast<double>(scores[0]);
    double b = std::exp(-a);
    objective += a + std::log1p(b + s);

    // Coefficients for the gradient
    std::for_each(ifirst, ikth, [&](size_type ix){ scratch[ix] = 0; });
    double coeff = 1 / (1 + s + b);
    scratch[idx[k]] = static_cast<Data>(coeff); // exp(kth - gt) / (1 + Z)
    scratch[label] = static_cast<Data>(- (1 + s) * coeff); // - Z / (1 + Z)
    for (size_type ix = k + 1; ix < num_tasks; ++ix) {
      scratch[idx[ix]] = scores[ix] * static_cast<Data>(coeff);
    }

    // Rank-1 update of the gradient
    sdca_blas_ger(D, T, static_cast<Data>(C), x_i, &scratch[0], grad);
  }

  objective *= C;
  objective += 0.5 * static_cast<double>(sdca_blas_dot(D * T, W, W));
  return objective;
}

template <typename Data>
inline double
eval_topk_softmax_nonconvex(
    const size_type num_dimensions,
    const size_type num_examples,
    const size_type num_tasks,
    const Data* X,
    const std::vector<size_type>& Y,
    const double C,
    const size_type k,
    const Data* W,
    std::vector<Data>& scores
    ) {
  auto first = scores.begin() + 1; // ground truth at 0, start at 1
  auto kth = scores.begin() + k; // first + k - 1
  auto last = scores.end();
  const blas_int D = static_cast<blas_int>(num_dimensions);
  const blas_int T = static_cast<blas_int>(num_tasks);

  double objective(0);
  for (size_type i = 0; i < num_examples; ++i) {
    const Data* x_i = X + num_dimensions * i;
    const size_type label = Y[i];

    // Compute scores
    sdca_blas_gemv(D, T, W, x_i, &scores[0], CblasTrans);
    std::swap(scores[0], scores[label]);

    // Find k largest elements
    std::nth_element(first, kth, last, std::greater<Data>());

    // Subtract the ground truth score for the T-k remaining tasks
    Data gt(scores[0]);
    std::for_each(kth, last, [=](Data &x){ x -= gt; });

    // Compute the log(1 + sum exp) loss
    objective += log_1_sum_exp(kth, last);
  }

  objective *= C;
  objective += 0.5 * static_cast<double>(sdca_blas_dot(D * T, W, W));
  return objective;
}

inline void
log_info_status(
    const std::string message,
    const size_type epoch,
    const double objective,
    const double optimality,
    const double wall_time,
    const double cpu_time
  ) {
  LOG_INFO << message << std::endl <<
    "epoch: " << std::setw(3) << epoch << std::setw(0) << ", "
    "objective: " << objective << ", "
    "optimality: " << optimality << ", "
    "wall_time: " << wall_time << ", "
    "cpu_time: " << cpu_time << std::endl;
}

inline void
log_verbose_progress(
    const size_type epoch,
    const size_type fun_evals,
    const double step_size,
    const double objective,
    const double optimality,
    const double wall_time,
    const double cpu_time
  ) {
  LOG_VERBOSE << "  "
    "epoch: " << std::setw(3) << epoch << std::setw(0) << ", "
    "fun_evals: " << std::setw(3) << fun_evals << std::setw(0) << ", "
    "step_size: " << step_size << ", "
    "objective: " << objective << ", "
    "optimality: " << optimality << ", "
    "wall_time: " << wall_time << ", "
    "cpu_time: " << cpu_time << std::endl;
}

template <typename Data>
inline bool
check_stopping_conditions(
    const solver_context<Data>& context,
    const size_type epoch,
    const size_type fun_evals,
    const double step_size,
    const double primal,
    const double optimality,
    stopwatch_wall& wall,
    stopwatch_cpu& cpu
  ) {
  wall.stop(); cpu.stop();
  log_verbose_progress(epoch, fun_evals, step_size, primal, optimality,
    wall.elapsed, cpu.elapsed);

  // Check stopping criteria
  if (optimality <= context.criteria.epsilon) {
    log_info_status("First order optimality condition is met.",
      epoch, primal, optimality, wall.elapsed, cpu.elapsed);
    return true;
  } else if (epoch >= context.criteria.max_epoch) {
    log_info_status("Maximum number of epochs exceeded.",
      epoch, primal, optimality, wall.elapsed, cpu.elapsed);
    return true;
  } else if (context.criteria.max_cpu_time > 0 &&
      cpu.elapsed >= context.criteria.max_cpu_time) {
    log_info_status("Maximum CPU time exceeded.",
      epoch, primal, optimality, wall.elapsed, cpu.elapsed);
    return true;
  } else if (context.criteria.max_wall_time > 0 &&
      wall.elapsed >= context.criteria.max_wall_time) {
    log_info_status("Maximum wall clock time exceeded.",
      epoch, primal, optimality, wall.elapsed, cpu.elapsed);
    return true;
  }
  wall.resume(); cpu.resume();
  return false;
}

/**
 * Optimize
 *    F(W) = C * \sum_i f_i(W, x_i, y_i) + 0.5 * ||W||_F^2,
 * where
 *    f_i(W,x,y) = \log(1 + \sum_{j = k, [j] \neq y}^T \exp(a_{[j]})),
 *    a_j = w_j^T x - w_y^T x,
 *    a_{[1]} \geq ... \geq a_{[T]}.
 * I.e. the inner sum goes over all tasks, except the ground truth
 * and the ones with the k-1 largest scores (these are ignored in the loss).
 **/
template <typename Data>
inline void
min_topk_softmax_nonconvex(
    const solver_context<Data>& context,
    const double C,
    const size_type k,
    Data* grad,
    Data* W_tmp,
    double& primal,
    double& optimality
    ) {
  // Input data
  dataset<Data> trn_data = context.datasets.front();
  const size_type num_dimensions = trn_data.num_dimensions;
  const size_type num_examples = trn_data.num_examples;
  const size_type num_tasks = trn_data.num_tasks;
  const Data* X = trn_data.data;
  const std::vector<size_type>& Y = trn_data.labels;
  const blas_int DT = static_cast<blas_int>(num_dimensions * num_tasks);
  Data* W = context.primal_variables;

  // Temporary memory
  std::vector<Data> scores(num_tasks);
  std::vector<Data> scratch(num_tasks);
  std::vector<size_type> idx(num_tasks);

  // Optimization variables
  size_type epoch(0), fun_evals(0);
  double step_size(0), min_step(1e-9), suff_decrease(1e-5), primal_tmp(0);
  stopwatch_wall wall;
  stopwatch_cpu cpu;
  wall.start(); cpu.start();

  // Evaluate initial point
  primal = grad_topk_softmax_nonconvex(
    num_dimensions, num_examples, num_tasks, X, Y, C, k, W, grad,
    scores, scratch, idx);
  ++fun_evals;
  optimality = sdca_blas_nrm2(DT, grad);

  if (check_stopping_conditions(context, epoch, fun_evals, step_size,
        primal, optimality, wall, cpu)) return;

  double step_size_before(0);
  step_size = std::max(min_step, std::min(1.0, 1 / optimality));
  for (;;) {
    ++epoch;
    if (step_size == step_size_before) {
      step_size *= 2.0;
    }
    step_size_before = step_size;

    // Armijo line search
    double coeff = suff_decrease * optimality; // * optimality;
    for (;;) {
      sdca_blas_copy(DT, W, W_tmp);
      sdca_blas_axpy(DT, static_cast<Data>(-step_size), grad, W_tmp);
      primal_tmp = eval_topk_softmax_nonconvex(
        num_dimensions, num_examples, num_tasks, X, Y, C, k, W_tmp, scores);
      ++fun_evals;
      if (primal_tmp <= primal - step_size * coeff) {
        break;
      }
      step_size *= 0.5;
      if (step_size < min_step) {
        log_info_status("Line search failed.",
          epoch, primal, optimality, wall.elapsed_now(), cpu.elapsed_now());
        return;
      }
    }

    // Compute objective value and the gradient at a new point
    sdca_blas_copy(DT, W_tmp, W);
    primal = grad_topk_softmax_nonconvex(
      num_dimensions, num_examples, num_tasks, X, Y, C, k, W, grad,
      scores, scratch, idx);
    ++fun_evals;
    optimality = sdca_blas_nrm2(DT, grad);

    if (check_stopping_conditions(context, epoch, fun_evals, step_size,
          primal, optimality, wall, cpu)) return;
  }
}

template <typename Data>
inline void
mex_main(
    const mxArray* data,
    const mxArray* labels,
    const mxArray* opts,
    mxArray* plhs[]
    ) {
  model_info<mxArray*> info;
  solver_context<Data> context;
  context.is_dual = false; // dual version (i.e. Gram matrix) is not supported
  set_datasets(data, labels, context);
  set_stopping_criteria(opts, context);

  auto trn_data = context.datasets.front();

  mxArray *mxW = mxDuplicateFieldOrCreateMatrix(opts, "W",
    trn_data.num_dimensions, trn_data.num_tasks, mex_class<Data>::id());
  mxCheckCreated(mxW, "W");
  context.primal_variables = static_cast<Data*>(mxGetData(mxW));
  info.add("W", mxW);

  mxArray* mxGrad = mxCreateNumericMatrix(trn_data.num_dimensions,
    trn_data.num_tasks, mex_class<Data>::id(), mxREAL);
  mxCheckCreated(mxGrad, "gradient");
  info.add("grad", mxGrad);

  mxArray* mxWTmp = mxCreateNumericMatrix(trn_data.num_dimensions,
    trn_data.num_tasks, mex_class<Data>::id(), mxREAL);
  mxCheckCreated(mxWTmp, "W_tmp");

  info.add("is_dual", mxCreateScalar(context.is_dual));
  info.add("num_dimensions", mxCreateScalar(trn_data.num_dimensions));
  info.add("num_examples", mxCreateScalar(trn_data.num_examples));
  info.add("num_tasks", mxCreateScalar(trn_data.num_tasks));

  std::string objective = mxGetFieldValueOrDefault(
    opts, "objective", std::string("l2_topk_softmax_nonconvex"));
  info.add("objective", mxCreateString(objective.c_str()));

  double c = mxGetFieldValueOrDefault<double>(opts, "c", 1);
  mxCheck<double>(std::greater<double>(), c, 0, "c");

  double num_examples(static_cast<double>(trn_data.num_examples));
  double C = mxGetFieldValueOrDefault<double>(opts, "C", c / num_examples);
  mxCheck<double>(std::greater<double>(), C, 0, "C");

  c = (C != c / num_examples) ? C * num_examples : c;
  info.add("c", mxCreateScalar(c));
  info.add("C", mxCreateScalar(C));

  size_type k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, trn_data.num_tasks - 1, "k");
  info.add("k", mxCreateScalar(k));

  double primal(0), optimality(0);
  if (objective == "l2_topk_softmax_nonconvex") {
    min_topk_softmax_nonconvex(context, C, k,
      static_cast<Data*>(mxGetData(mxGrad)),
      static_cast<Data*>(mxGetData(mxWTmp)),
      primal, optimality);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_objective], err_msg[err_objective], objective.c_str());
  }

  info.add("primal", mxCreateScalar(primal));
  info.add("optimality", mxCreateScalar(optimality));
  info.add("max_epoch", mxCreateScalar(context.criteria.max_epoch));
  info.add("max_cpu_time", mxCreateScalar(context.criteria.max_cpu_time));
  info.add("max_wall_time", mxCreateScalar(context.criteria.max_wall_time));
  info.add("epsilon", mxCreateScalar(context.criteria.epsilon));
  info.add("log_level", mxCreateString(logging::get_level_name()));
  info.add("log_format", mxCreateString(logging::get_format_name()));
  info.add("precision", mxCreateString(type_traits<double>::name()));
  info.add("data_precision", mxCreateString(type_traits<Data>::name()));
  plhs[0] = mxCreateStruct(info.fields, "model");;
}

void
mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 3, 3, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);

  const mxArray* data = prhs[0];
  const mxArray* labels = prhs[1];
  const mxArray* opts = prhs[2];
  mxCheckStruct(opts, "opts");

  logging::format_push();
  mat_cout_hijack mat_cout;
  set_logging_options(opts);
  if (mxIsDouble(data) || (mxIsCell(data) && !mxIsEmpty(data)
      && mxIsDouble(mxGetCell(data, 0)))) {
     mex_main<double>(data, labels, opts, plhs);
  } else if (mxIsSingle(data) || (mxIsCell(data) && !mxIsEmpty(data)
      && mxIsSingle(mxGetCell(data, 0)))) {
     mex_main<float>(data, labels, opts, plhs);
  } else {
    mexErrMsgIdAndTxt(err_id[err_arg], err_msg[err_arg]);
  }
  mat_cout.release();
  logging::format_pop();
}

#ifndef SDCA_MATLAB_MEX_UTIL_H
#define SDCA_MATLAB_MEX_UTIL_H

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include <mex.h>

#include "solve/solvedef.h"
#include "util/logging.h"

namespace sdca {

enum err_index {
  err_arg = 0,
  err_arg_count,
  err_arg_single,
  err_arg_double,
  err_arg_real,
  err_arg_square,
  err_arg_struct,
  err_arg_not_sparse,
  err_arg_not_empty,
  err_arg_range,
  err_arg_class,
  err_vec_dim_class,
  err_mat_dim_class,
  err_out_of_memory,
  err_read_failed,
  err_labels_range,
  err_cell_arrays,
  err_num_dim,
  err_num_tasks,
  err_num_ex,
  err_command,
  err_prox,
  err_objective,
  err_summation,
  err_precision,
  err_log_level,
  err_log_format,
  err_not_implemented
};

static const char* err_id[] = {
  "LIBSDCA:arg",
  "LIBSDCA:arg_count",
  "LIBSDCA:arg_single",
  "LIBSDCA:arg_double",
  "LIBSDCA:arg_real",
  "LIBSDCA:arg_square",
  "LIBSDCA:arg_struct",
  "LIBSDCA:arg_not_sparse",
  "LIBSDCA:arg_not_empty",
  "LIBSDCA:arg_range",
  "LIBSDCA:arg_class",
  "LIBSDCA:vec_dim_class",
  "LIBSDCA:mat_dim_class",
  "LIBSDCA:out_of_memory",
  "LIBSDCA:read_failed",
  "LIBSDCA:labels_range",
  "LIBSDCA:cell_arrays",
  "LIBSDCA:num_dim",
  "LIBSDCA:num_tasks",
  "LIBSDCA:num_examples",
  "LIBSDCA:command",
  "LIBSDCA:prox",
  "LIBSDCA:objective",
  "LIBSDCA:summation",
  "LIBSDCA:precision",
  "LIBSDCA:log_level",
  "LIBSDCA:log_format",
  "LIBSDCA:not_implemented"
};

static const char* err_msg[] = {
  "Invalid input.",
  "Invalid number of input/output arguments.",
  "'%s' must be single.",
  "'%s' must be double.",
  "'%s' must be single or double.",
  "'%s' must be a square matrix.",
  "'%s' must be a struct.",
  "'%s' must be not sparse.",
  "'%s' must be not empty.",
  "'%s' is out of range.",
  "'%s' must be of class %s.",
  "'%s' must be a %u dimensional vector of class %s.",
  "'%s' must be a %u-by-%u dimensional matrix of class %s.",
  "Out of memory (cannot allocate memory for '%s').",
  "Failed to read the value of '%s'.",
  "Invalid labels range (must be 1:T or 0:T-1).",
  "Invalid cell arrays (must be non-empty and have the same size).",
  "Invalid number of dimensions in features #%d.",
  "Invalid number of tasks in labels #%d.",
  "Invalid number of training examples (first dimension) in kernel #%d.",
  "Unknown command '%s'.",
  "Unknown prox '%s'.",
  "Unknown objective '%s'.",
  "Unknown summation '%s'.",
  "Unknown precision '%s'.",
  "Unknown log_level '%s'.",
  "Unknown log_format '%s'.",
  "%s is not implemented yet."
};

inline const char*
to_string(mxClassID class_id) {
  switch (class_id) {
    case mxDOUBLE_CLASS: return "double";
    case mxSINGLE_CLASS: return "single";
    default: return "unknown";
  }
}

template <typename Type>
struct mex_class {
  static constexpr mxClassID
  id() { return mxUNKNOWN_CLASS; }
};

template <>
struct mex_class<float> {
  static constexpr mxClassID
  id() { return mxSINGLE_CLASS; }
};

template <>
struct mex_class<double> {
  static constexpr mxClassID
  id() { return mxDOUBLE_CLASS; }
};

template <typename Usage>
inline void
mxCheckArgNum(
    const int argnum,
    const int min,
    const int max,
    Usage usage
    ) {
  if (argnum < min || argnum > max) {
    usage();
    mexErrMsgIdAndTxt(err_id[err_arg_count], err_msg[err_arg_count]);
  }
}

template <typename Type>
inline void
mxCheckRange(
    const Type var,
    const Type min,
    const Type max,
    const char* name
    ) {
  if (var < min || var > max) {
    mexErrMsgIdAndTxt(err_id[err_arg_range], err_msg[err_arg_range], name);
  }
}

template <typename Type, typename Compare>
inline void
mxCheck(
    Compare comp,
    const Type var,
    const Type value,
    const char* name
    ) {
  if (!comp(var, value)) {
    mexErrMsgIdAndTxt(err_id[err_arg_range], err_msg[err_arg_range], name);
  }
}

inline void
mxCheckSingle(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsSingle(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_single], err_msg[err_arg_single], name);
  }
}

inline void
mxCheckDouble(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsDouble(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_double], err_msg[err_arg_double], name);
  }
}

inline void
mxCheckClass(
    const mxArray* pa,
    const char* name,
    const mxClassID class_id
    ) {
  if (pa != nullptr && mxGetClassID(pa) != class_id) {
    mexErrMsgIdAndTxt(err_id[err_arg_class], err_msg[err_arg_class], name,
      to_string(class_id));
  }
}

inline void
mxCheckReal(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && (mxIsComplex(pa) ||
     (!mxIsSingle(pa) && !mxIsDouble(pa)))) {
    mexErrMsgIdAndTxt(err_id[err_arg_real], err_msg[err_arg_real], name);
  }
}

inline void
mxCheckSquare(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && (mxGetM(pa) != mxGetN(pa))) {
    mexErrMsgIdAndTxt(err_id[err_arg_square], err_msg[err_arg_square], name);
  }
}

inline void
mxCheckCellArrays(
    const mxArray* pa,
    const mxArray* pb
    ) {
  if (pa == nullptr || pb == nullptr
      || mxIsEmpty(pa) || mxIsEmpty(pb)
      || !mxIsCell(pa) || !mxIsCell(pb)
      || mxGetNumberOfElements(pa) != mxGetNumberOfElements(pb)) {
    mexErrMsgIdAndTxt(err_id[err_cell_arrays], err_msg[err_cell_arrays]);
  }
}

inline void
mxCheckStruct(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsStruct(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_struct], err_msg[err_arg_struct], name);
  }
}

inline void
mxCheckNotSparse(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && mxIsSparse(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_arg_not_sparse], err_msg[err_arg_not_sparse], name);
  }
}

inline void
mxCheckNotEmpty(
    const mxArray* pa,
    const char* name
    ) {
  if (pa == nullptr || mxIsEmpty(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_arg_not_empty], err_msg[err_arg_not_empty], name);
  }
}

inline void
mxCheckCreated(
    const mxArray* pa,
    const char* name
    ) {
  if (pa == nullptr) {
    mexErrMsgIdAndTxt(
      err_id[err_out_of_memory], err_msg[err_out_of_memory], name);
  }
}

template <typename SizeType>
inline void
mxCheckVector(
    const mxArray* pa,
    const char* name,
    const SizeType n,
    const mxClassID class_id = mxDOUBLE_CLASS
    ) {
  if (!( (static_cast<SizeType>(mxGetM(pa)) == n && mxGetN(pa) == 1) ||
         (mxGetM(pa) == 1 && static_cast<SizeType>(mxGetN(pa)) == n) ) ||
      mxIsComplex(pa) || mxGetClassID(pa) != class_id) {
    mexErrMsgIdAndTxt(
      err_id[err_vec_dim_class], err_msg[err_vec_dim_class], name,
      static_cast<std::size_t>(n), to_string(class_id));
  }
}

template <typename SizeType>
inline void
mxCheckMatrix(
    const mxArray* pa,
    const char* name,
    const SizeType m,
    const SizeType n,
    const mxClassID class_id = mxDOUBLE_CLASS
    ) {
  if (static_cast<SizeType>(mxGetM(pa)) != m ||
      static_cast<SizeType>(mxGetN(pa)) != n ||
      mxIsComplex(pa) || mxGetClassID(pa) != class_id) {
    mexErrMsgIdAndTxt(
      err_id[err_mat_dim_class], err_msg[err_mat_dim_class], name,
      static_cast<std::size_t>(m), static_cast<std::size_t>(n),
      to_string(class_id));
  }
}

template <typename Type>
inline const Type
mxGetFieldValueOrDefault(
    const mxArray* pa,
    const char* name,
    const Type value
    ) {
  if (pa != nullptr) {
    mxArray* field = mxGetField(pa, 0, name);
    if (field != nullptr) {
      return static_cast<Type>(mxGetScalar(field));
    }
  }
  return value;
}

inline const std::string
mxGetString(
    const mxArray* pa,
    const char* name
    ) {
  mwSize buflen = mxGetNumberOfElements(pa) + 1;
  char* buf = static_cast<char*>(mxCalloc(buflen, sizeof(char)));
  if (mxGetString(pa, buf, buflen) == 0) {
    return std::string(buf);
  }
  mexErrMsgIdAndTxt(
    err_id[err_read_failed], err_msg[err_read_failed], name);
  return nullptr;
}

template <>
inline const std::string
mxGetFieldValueOrDefault(
    const mxArray* pa,
    const char* name,
    const std::string value
    ) {
  if (pa != nullptr) {
    mxArray* field = mxGetField(pa, 0, name);
    if (field != nullptr) {
      return mxGetString(field, name);
    }
  }
  return value;
}

template <typename Type>
inline void
mxSetFieldValue(
    const mxArray* pa,
    const char* name,
    Type& value
    ) {
  if (pa != nullptr) {
    mxArray* field = mxGetField(pa, 0, name);
    if (field != nullptr) {
      value = static_cast<Type>(mxGetScalar(field));
    }
  }
}

template <typename Type>
inline mxArray*
mxCreateScalar(
    const Type value
  ) {
  return mxCreateDoubleScalar(static_cast<double>(value));
}

template <typename Type>
inline mxArray*
mxCreateVector(
    const std::vector<Type> vec,
    const char* name
  ) {
  mxArray* pa = mxCreateDoubleMatrix(vec.size(), 1, mxREAL);
  mxCheckCreated(pa, name);
  double* data = mxGetPr(pa);
  for (Type v : vec) {
    *data++ = static_cast<double>(v);
  }
  return pa;
}

inline mxArray*
mxCreateStruct(
    const std::vector<std::pair<const char*, mxArray*>>& fields,
    const char* name
  ) {
  std::vector<const char*> names;
  names.reserve(fields.size());
  for (auto field : fields) {
    names.push_back(field.first);
  }
  mxArray* pa = mxCreateStructMatrix(1, 1,
    static_cast<int>(fields.size()), &names[0]);
  mxCheckCreated(pa, name);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    mxSetFieldByNumber(pa, 0, static_cast<int>(i), fields[i].second);
  }
  return pa;
}

inline mxArray*
mxDuplicateFieldOrCreateMatrix(
    const mxArray* structure,
    const char* field,
    const std::size_t m,
    const std::size_t n,
    const mxClassID id
  ) {
  mxArray *pa = mxGetField(structure, 0, field);
  if (pa != nullptr) {
    mxCheckMatrix(pa, field, m, n, id);
    pa = mxDuplicateArray(pa);
  } else {
    pa = mxCreateNumericMatrix(m, n, id, mxREAL);
  }
  mxCheckCreated(pa, field);
  return pa;
}

/**
 * Helper methods for solvers.
 **/

template <typename Data>
inline void
set_stopping_criteria(
    const mxArray* opts,
    solver_context<Data>& context
  ) {
  auto c = &context.criteria;
  mxSetFieldValue(opts, "check_on_start", c->check_on_start);
  mxSetFieldValue(opts, "check_epoch", c->check_epoch);
  mxSetFieldValue(opts, "max_epoch", c->max_epoch);
  mxSetFieldValue(opts, "max_cpu_time", c->max_cpu_time);
  mxSetFieldValue(opts, "max_wall_time", c->max_wall_time);
  mxSetFieldValue(opts, "epsilon", c->epsilon);
  mxCheck<size_type>(std::greater_equal<size_type>(),
    c->check_epoch, 0, "check_epoch");
  mxCheck<size_type>(std::greater_equal<size_type>(),
    c->max_epoch, 0, "max_epoch");
  mxCheck<double>(std::greater_equal<double>(),
    c->max_cpu_time, 0, "max_cpu_time");
  mxCheck<double>(std::greater_equal<double>(),
    c->max_wall_time, 0, "max_wall_time");
  mxCheck<double>(std::greater_equal<double>(),
    c->epsilon, 0, "epsilon");
}

template <typename Data>
inline void
set_labels(
    const mxArray* labels,
    dataset<Data>& data_set
  ) {
  size_type n = data_set.num_examples;
  mxCheckVector(labels, "labels", n);

  std::vector<size_type> vec(mxGetPr(labels), mxGetPr(labels) + n);
  auto minmax = std::minmax_element(vec.begin(), vec.end());
  if (*minmax.first == 1) {
    std::for_each(vec.begin(), vec.end(), [](size_type &x){ x -= 1; });
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(err_id[err_labels_range], err_msg[err_labels_range]);
  }

  data_set.labels = std::move(vec);
  data_set.num_tasks = static_cast<size_type>(*minmax.second) + 1;
}

template <typename Data>
inline void
set_dataset(
    const mxArray* data,
    const mxArray* labels,
    solver_context<Data>& context
  ) {
  mxCheckNotSparse(data, "data");
  mxCheckNotEmpty(data, "data");
  mxCheckReal(data, "data");
  mxCheckClass(data, "data", mex_class<Data>::id());

  mxCheckNotSparse(labels, "labels");
  mxCheckNotEmpty(labels, "labels");
  mxCheckDouble(labels, "labels");

  dataset<Data> data_set;
  data_set.data = static_cast<Data*>(mxGetData(data));
  data_set.num_dimensions = (context.is_dual) ? 0 : mxGetM(data);
  data_set.num_examples = mxGetN(data);
  set_labels(labels, data_set);

  context.datasets.emplace_back(data_set);
}

template <typename Data>
inline void
set_datasets(
    const mxArray* data,
    const mxArray* labels,
    solver_context<Data>& context
  ) {
  if (mxIsNumeric(data)) {
    if (context.is_dual) {
      mxCheckSquare(data, "data");
    }
    set_dataset(data, labels, context);
  } else {
    mxCheckCellArrays(data, labels);
    if (context.is_dual) {
      mxCheckSquare(mxGetCell(data, 0), "data");
    }

    // Training dataset
    set_dataset(mxGetCell(data, 0), mxGetCell(labels, 0), context);
    size_type num_dimensions = context.datasets[0].num_dimensions;
    size_type num_examples = context.datasets[0].num_examples;
    size_type num_tasks = context.datasets[0].num_tasks;

    // Testing datasets
    size_type num_datasets = mxGetNumberOfElements(data);
    for (size_type i = 1; i < num_datasets; ++i) {
      set_dataset(mxGetCell(data, i), mxGetCell(labels, i), context);
      if (num_dimensions != context.datasets[i].num_dimensions) {
        mexErrMsgIdAndTxt(err_id[err_num_dim], err_msg[err_num_dim], i + 1);
      }
      if (num_tasks != context.datasets[i].num_tasks) {
        mexErrMsgIdAndTxt(err_id[err_num_tasks], err_msg[err_num_tasks], i + 1);
      }
      if (context.is_dual && num_examples != mxGetM(mxGetCell(data, i))) {
        mexErrMsgIdAndTxt(err_id[err_num_ex], err_msg[err_num_ex], i + 1);
      }
    }
  }
}

/**
 * Logging in Matlab.
 **/

inline void
set_logging_options(
    const mxArray* opts
  ) {
  std::string log_level = mxGetFieldValueOrDefault(
    opts, "log_level", std::string("info"));
  if (log_level == "none") {
    logging::set_level(logging::none);
  } else if (log_level == "info") {
    logging::set_level(logging::info);
  } else if (log_level == "verbose") {
    logging::set_level(logging::verbose);
  } else if (log_level == "debug") {
    logging::set_level(logging::debug);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_log_level], err_msg[err_log_level], log_level.c_str());
  }

  std::string log_format = mxGetFieldValueOrDefault(
    opts, "log_format", std::string("short_e"));
  if (log_format == "short_f") {
    logging::set_format(logging::short_f);
  } else if (log_format == "short_e") {
    logging::set_format(logging::short_e);
  } else if (log_format == "long_f") {
    logging::set_format(logging::long_f);
  } else if (log_format == "long_e") {
    logging::set_format(logging::long_e);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_log_format], err_msg[err_log_format], log_format.c_str());
  }
}

// http://stackoverflow.com/questions/243696/correctly-over-loading-a-stringbuf-to-replace-cout-in-a-matlab-mex-file
class mat_streambuf : public std::streambuf {
protected:
  virtual std::streamsize xsputn(const char_type* s, std::streamsize n) {
    mexPrintf("%.*s", n, s);
    mexEvalString("drawnow;");
    return n;
  }

  virtual int_type overflow(int_type c = traits_type::eof()) {
    if (c != traits_type::eof()) {
      mexPrintf("%c", c);
      mexEvalString("drawnow;");
    }
    return c;
  }
};

class mat_cout_hijack {
public:
  mat_cout_hijack() {
    std_buf_ = std::cout.rdbuf(&mat_buf_);
  }
  ~mat_cout_hijack() {
    release();
  }
  void release() {
    std::cout.rdbuf(std_buf_);
  }
private:
  mat_streambuf mat_buf_;
  std::streambuf* std_buf_;
};

}

#endif

#ifndef SDCA_MATLAB_MEX_UTIL_H
#define SDCA_MATLAB_MEX_UTIL_H

// STL
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

// Matlab
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

#include <mex.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// SDCA
#include "sdca/solver/solverdef.h"
#include "sdca/utility/logging.h"

namespace sdca {

//-----------------------------------------------------------------------------
// Type names
//-----------------------------------------------------------------------------

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


//-----------------------------------------------------------------------------
// Errors and messages
//-----------------------------------------------------------------------------


enum err_index {
  err_exception = 0,
  err_arg,
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
  err_num_classes,
  err_num_examples,
  err_command,
  err_prox,
  err_objective,
  err_summation,
  err_precision,
  err_log_level,
  err_log_format,
  err_help_arg,
  err_not_implemented
};

static const char* err_id[] = {
  "LIBSDCA:exception",
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
  "LIBSDCA:num_classes",
  "LIBSDCA:num_examples",
  "LIBSDCA:command",
  "LIBSDCA:prox",
  "LIBSDCA:objective",
  "LIBSDCA:summation",
  "LIBSDCA:precision",
  "LIBSDCA:log_level",
  "LIBSDCA:log_format",
  "LIBSDCA:help_arg",
  "LIBSDCA:not_implemented"
};

static const char* err_msg[] = {
  "Exception: %s.",
  "Invalid input.",
  "Invalid number of input or output arguments.",
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
  "Invalid number of classes in labels #%d.",
  "Invalid number of training examples (first dimension) in kernel #%d.",
  "Unknown command '%s'.",
  "Unknown prox '%s'.",
  "Unknown objective '%s'.",
  "Unknown summation '%s'.",
  "Unknown precision '%s'.",
  "Unknown log_level '%s'.",
  "Unknown log_format '%s'.",
  "Unknown help argument '%s'.",
  "%s is not implemented yet."
};


//-----------------------------------------------------------------------------
// mxCheck* methods
//-----------------------------------------------------------------------------

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


//-----------------------------------------------------------------------------
// mxGet* methods
//-----------------------------------------------------------------------------

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


//-----------------------------------------------------------------------------
// mxSet* methods
//-----------------------------------------------------------------------------

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


//-----------------------------------------------------------------------------
// mxCreate* methods
//-----------------------------------------------------------------------------

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
    const mxArray* pa,
    const char* field,
    const std::size_t m,
    const std::size_t n,
    const mxClassID id
  ) {
  mxArray *pb = (pa != nullptr) ? mxGetField(pa, 0, field) : nullptr;
  if (pb != nullptr) {
    mxCheckMatrix(pb, field, m, n, id);
    pb = mxDuplicateArray(pb);
  } else {
    pb = mxCreateNumericMatrix(m, n, id, mxREAL);
  }
  mxCheckCreated(pb, field);
  return pb;
}


//-----------------------------------------------------------------------------
// Helper methods and structs for solvers
//-----------------------------------------------------------------------------

template <typename Field>
struct model_info {
  typedef Field field_type;
  std::vector<std::pair<const char*, field_type>> fields;

  inline void add(
      const char* name,
      const field_type value) {
    fields.emplace_back(std::make_pair(name, value));
  }
};

//-----------------------------------------------------------------------------
// Logging in Matlab
//-----------------------------------------------------------------------------

inline void
set_logging_options(
    const mxArray* opts
  ) {
  std::string log_level = mxGetFieldValueOrDefault(
    opts, "log_level", std::string("info"));
  if (log_level == "none") {
    logging::set_level(logging::level::none);
  } else if (log_level == "warning") {
    logging::set_level(logging::level::warning);
  } else if (log_level == "info") {
    logging::set_level(logging::level::info);
  } else if (log_level == "verbose") {
    logging::set_level(logging::level::verbose);
  } else if (log_level == "debug") {
    logging::set_level(logging::level::debug);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_log_level], err_msg[err_log_level], log_level.c_str());
  }

  std::string log_format = mxGetFieldValueOrDefault(
    opts, "log_format", std::string("short_e"));
  if (log_format == "short_e") {
    logging::set_format(logging::format::short_e);
  } else if (log_format == "short_f") {
    logging::set_format(logging::format::short_f);
  } else if (log_format == "long_e") {
    logging::set_format(logging::format::long_e);
  } else if (log_format == "long_f") {
    logging::set_format(logging::format::long_f);
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

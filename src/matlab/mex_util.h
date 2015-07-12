#ifndef SDCA_MATLAB_MEX_UTIL_H
#define SDCA_MATLAB_MEX_UTIL_H

#include <string>
#include <vector>
#include <utility>
#include <mex.h>

namespace sdca {

enum err_index {
  err_argnum = 0,
  err_arg_single,
  err_arg_double,
  err_arg_real,
  err_arg_square,
  err_arg_struct,
  err_arg_not_sparse,
  err_arg_not_empty,
  err_arg_range,
  err_vec_dim,
  err_out_of_memory,
  err_read_failed,
  err_labels_range,
  err_proj_type,
  err_obj_type,
  err_not_implemented
};

static const char* err_id[] = {
  "LIBSDCA:argnum",
  "LIBSDCA:arg_single",
  "LIBSDCA:arg_double",
  "LIBSDCA:arg_real",
  "LIBSDCA:arg_square",
  "LIBSDCA:arg_struct",
  "LIBSDCA:arg_not_sparse",
  "LIBSDCA:arg_not_empty",
  "LIBSDCA:arg_range",
  "LIBSDCA:vec_dim",
  "LIBSDCA:out_of_memory",
  "LIBSDCA:read_failed",
  "LIBSDCA:labels_range",
  "LIBSDCA:proj_type",
  "LIBSDCA:obj_type",
  "LIBSDCA:not_implemented"
};

static const char* err_msg[] = {
  "Invalid number of input/output arguments.",
  "'%s' must be single.",
  "'%s' must be double.",
  "'%s' must be single or double.",
  "'%s' must be a square matrix.",
  "'%s' must be a struct.",
  "'%s' must be not sparse.",
  "'%s' must be not empty.",
  "'%s' is out of range.",
  "'%s' must be a %u dimensional vector.",
  "Out of memory (cannot allocate memory for '%s').",
  "Failed to read the value of '%s'.",
  "Invalid labels range (must be 1:T or 0:T-1).",
  "Unknown projection type '%s'.",
  "Unknown objective type '%s'.",
  "%s is not implemented yet."
};

template <typename Usage>
void
mxCheckArgNum(
    const int argnum,
    const int min,
    const int max,
    Usage usage
    ) {
  if (argnum < min || argnum > max) {
    usage();
    mexErrMsgIdAndTxt(err_id[err_argnum], err_msg[err_argnum]);
  }
}

template <typename Type>
void
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
void
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

void
mxCheckSingle(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsSingle(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_single], err_msg[err_arg_single], name);
  }
}

void
mxCheckDouble(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsDouble(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_double], err_msg[err_arg_double], name);
  }
}

void
mxCheckReal(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && (mxIsComplex(pa) ||
     (!mxIsSingle(pa) && !mxIsDouble(pa)))) {
    mexErrMsgIdAndTxt(err_id[err_arg_real], err_msg[err_arg_real], name);
  }
}

void
mxCheckSquare(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && (mxGetM(pa) != mxGetN(pa))) {
    mexErrMsgIdAndTxt(err_id[err_arg_square], err_msg[err_arg_square], name);
  }
}

void
mxCheckStruct(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsStruct(pa)) {
    mexErrMsgIdAndTxt(err_id[err_arg_struct], err_msg[err_arg_struct], name);
  }
}

void
mxCheckNotSparse(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && mxIsSparse(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_arg_not_sparse], err_msg[err_arg_not_sparse], name);
  }
}

void
mxCheckNotEmpty(
    const mxArray* pa,
    const char* name
    ) {
  if (pa == nullptr || mxIsEmpty(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_arg_not_empty], err_msg[err_arg_not_empty], name);
  }
}

void
mxCheckCreated(
    const mxArray* pa,
    const char* name
    ) {
  if (pa == nullptr) {
    mexErrMsgIdAndTxt(
      err_id[err_out_of_memory], err_msg[err_out_of_memory], name);
  }
}

template <typename IndexType>
void
mxCheckVector(
    const IndexType dim,
    const mxArray* pa,
    const char* name
    ) {
  if (!( (static_cast<IndexType>(mxGetM(pa)) == dim && mxGetN(pa) == 1)
      || (mxGetM(pa) == 1 && static_cast<IndexType>(mxGetN(pa)) == dim) )) {
    mexErrMsgIdAndTxt(
      err_id[err_vec_dim], err_msg[err_vec_dim], name,
      static_cast<std::size_t>(dim));
  }
}


template <typename Type>
inline
const Type
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
inline
const std::string
mxGetFieldValueOrDefault(
    const mxArray* pa,
    const char* name,
    const std::string value
    ) {
  if (pa != nullptr) {
    mxArray* field = mxGetField(pa, 0, name);
    if (field != nullptr) {
      mwSize buflen = mxGetNumberOfElements(field) + 1;
      char* buf = static_cast<char*>(mxCalloc(buflen, sizeof(char)));
      if (mxGetString(field, buf, buflen) == 0) {
        return std::string(buf);
      }
      mexErrMsgIdAndTxt(
        err_id[err_read_failed], err_msg[err_read_failed], name);
    }
  }
  return value;
}

template <typename Type>
inline
void
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
inline
mxArray*
mxCreateScalar(
    const Type value
  ) {
  return mxCreateDoubleScalar(static_cast<double>(value));
}

inline
mxArray*
mxCreateStruct(
    const std::vector<std::pair<std::string, mxArray*>>& fields,
    const char* name
  ) {
  std::vector<const char*> names;
  names.reserve(fields.size());
  for (auto field : fields) {
    names.push_back(field.first.c_str());
  }
  mxArray* pa = mxCreateStructMatrix(1, 1,
    static_cast<int>(fields.size()), &names[0]);
  mxCheckCreated(pa, name);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    mxSetFieldByNumber(pa, 0, static_cast<int>(i), fields[i].second);
  }
  return pa;
}

}

#endif

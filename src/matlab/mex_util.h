#ifndef SDCA_MATLAB_MEX_UTIL_H
#define SDCA_MATLAB_MEX_UTIL_H

#include <string>
#include <mex.h>

namespace sdca {

enum error_index {
  err_argnum = 0,
  err_type_real,
  err_type_struct,
  err_type_no_sparse,
  err_var_range,
  err_out_of_memory,
  err_read_failed,
  err_proj_type
};

static const char* err_id[] = {
  "LIBSDCA:argnum",
  "LIBSDCA:type_real",
  "LIBSDCA:type_struct",
  "LIBSDCA:type_no_sparse",
  "LIBSDCA:var_range",
  "LIBSDCA:out_of_memory",
  "LIBSDCA:read_failed",
  "LIBSDCA:proj_type"
};

static const char* err_msg[] = {
  "Invalid number of input/output arguments.",
  "'%s' must be single or double.",
  "'%s' must be a struct.",
  "'%s' must not be sparse.",
  "'%s' is out of range.",
  "Out of memory (cannot allocate memory for '%s').",
  "Failed to read the value of '%s'.",
  "Unknown projection type '%s'."
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
    mexErrMsgIdAndTxt(err_id[err_var_range], err_msg[err_var_range], name);
  }
}

void
mxCheckReal(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsDouble(pa) && !mxIsSingle(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_type_real], err_msg[err_type_real], name);
  }
}

void
mxCheckStruct(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && !mxIsStruct(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_type_struct], err_msg[err_type_struct], name);
  }
}

void
mxCheckNotSparse(
    const mxArray* pa,
    const char* name
    ) {
  if (pa != nullptr && mxIsSparse(pa)) {
    mexErrMsgIdAndTxt(
      err_id[err_type_no_sparse], err_msg[err_type_no_sparse], name);
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

}

#endif

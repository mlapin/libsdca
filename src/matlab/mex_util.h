#ifndef SDCA_MATLAB_MEX_UTIL_H
#define SDCA_MATLAB_MEX_UTIL_H

#include <string>
#include <mex.h>

namespace sdca {

enum error_index {
  err_argnum_input = 0,
  err_argnum_output,
  err_argtype_real,
  err_argopt_struct,
  err_out_of_memory,
  err_read_fail,
};

static const char* err_id[] = {
  "LIBSDCA:argnum_input",
  "LIBSDCA:argnum_output",
  "LIBSDCA:argtype_real",
  "LIBSDCA:argopt_struct",
  "LIBSDCA:out_of_memory",
  "LIBSDCA:read_fail",
};

static const char* err_msg[] = {
  "Wrong number of input arguments.",
  "Wrong number of output arguments.",
  "Input data type must be either single or double.",
  "Options argument must be a struct.",
  "Out of memory (cannot allocate memory for '%s').",
  "Failed to read the value of '%s'.",
};

template <typename Type>
inline
const Type
mxGetFieldValueOrDefault(
    const mxArray* pa,
    const char* name,
    const Type value
    ) {
  mxArray* field = mxGetField(pa, 0, name);
  return field ? static_cast<Type>(mxGetScalar(field)) : value;
}

template <>
inline
const std::string
mxGetFieldValueOrDefault(
    const mxArray* pa,
    const char* name,
    const std::string value
    ) {
  mxArray* field = mxGetField(pa, 0, name);
  if (field) {
    mwSize buflen = mxGetNumberOfElements(field) + 1;
    char* buf = static_cast<char*>(mxCalloc(buflen, sizeof(char)));
    if (mxGetString(field, buf, buflen) == 0) {
      return std::string(buf, buflen);
    }
    mexErrMsgIdAndTxt(err_id[err_read_fail], err_msg[err_read_fail], name);
  }
  return value;
}

}

#endif

#ifndef SDCA_MEX_UTIL_HPP
#define SDCA_MEX_UTIL_HPP

#include <strings.h>
#include <cstddef>

#include "mex.h"

#include "common.hpp"

namespace sdca {

const char *errInvalidArgument = "SDCA:invalidArgument";
const char *errOutOfMemory = "SDCA:outOfMemory";
const char *errSolverError = "SDCA:solverError";

template <typename T>
mxArray *mxCreatePrecisionString();

template <>
mxArray *mxCreatePrecisionString<float>() {
  return mxCreateString("float");
}

template <>
mxArray *mxCreatePrecisionString<double>() {
  return mxCreateString("double");
}

mxArray *mxCreateScalar(SizeType x) {
  mxArray *array = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(array) = static_cast<double>(x);
  return array;
}

mxArray *mxCreateScalar(double x) {
  mxArray *array = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(array) = x;
  return array;
}

mxArray * createScalarStructArray(void const **fields) {
  void const **iter;
  char const **niter;
  char const **names;
  int numFields = 0;
  mxArray *s;
  mwSize dims [] = {1, 1};

  for (iter = fields; *iter; iter += 2) {
    numFields++;
  }

  names = static_cast<const char **>(
    mxCalloc((std::size_t) numFields, sizeof(char const*)));

  for (iter = fields, niter = names; *iter; iter += 2, niter++) {
    *niter = static_cast<const char *>(*iter);
  }

  s = mxCreateStructArray(2, dims, numFields, names);
  for (iter = fields, niter = names; *iter; iter += 2, niter++) {
    mxSetField(s, 0, *niter,
      const_cast<mxArray*>(reinterpret_cast<const mxArray*>(*(iter+1)))
      );
  }
  return s ;
}


SizeType *mxCreateLabelsVector(const mxArray *mxY,
    SizeType &min_label, SizeType &max_label) {
  if (!mxIsDouble(mxY)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "Y must be double.");
  }
  if (mxGetM(mxY) != 1 && mxGetN(mxY) != 1) {
    mexErrMsgIdAndTxt(errInvalidArgument, "Y must be a vector.");
  }

  size_t N = mxGetM(mxY) > mxGetN(mxY) ? mxGetM(mxY) : mxGetN(mxY);
  SizeType *Y = static_cast<SizeType *>(mxMalloc(N * sizeof(*Y)));
  if (Y == NULL) {
    mexErrMsgIdAndTxt(errOutOfMemory, "Failed to allocate memory for Y.");
  }

  min_label = 1;
  max_label = 0;
  double *y = mxGetPr(mxY);
  for (size_t i = 0; i < N; ++i) {
    if (y[i] < 1.0) {
      mexErrMsgIdAndTxt(errInvalidArgument, "Labels must be in the range 1:T.");
    } else {
      Y[i] = static_cast<SizeType>(y[i] - 1);
      if (Y[i] < min_label) min_label = Y[i];
      if (Y[i] > max_label) max_label = Y[i];
    }
  }
  if (min_label > 0) {
    mexErrMsgIdAndTxt(errInvalidArgument, "Labels must be in the range 1:T.");
  }

  return Y;
}

void mxVerifySparseNotEmpty(const mxArray *x, const char *name) {
  if (!mxIsSparse(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be sparse.", name);
  }
  if (mxIsEmpty(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be non-empty.", name);
  }
}

void mxVerifyNotSparseNotEmpty(const mxArray *x, const char *name) {
  if (mxIsSparse(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be full.", name);
  }
  if (mxIsEmpty(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be non-empty.", name);
  }
}

void mxVerifySingleOrDouble(const mxArray *x, const char *name) {
  if (!(mxIsSingle(x) || mxIsDouble(x))) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be single or double.", name);
  }
}

void mxVerifyDouble(const mxArray *x, const char *name) {
  if (!mxIsDouble(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be double.", name);
  }
}

void mxVerifyNumeric(const mxArray *x, const char *name) {
  if (!mxIsNumeric(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be numeric.", name);
  }
}

void mxVerifySameClass(const mxArray *x, const mxArray *y,
                       const char *nx, const char *ny) {
  if (mxGetClassID(x) != mxGetClassID(y)) {
    mexErrMsgIdAndTxt(errInvalidArgument,
      "%s and %s must be of the same type.", nx, ny);
  }
}

void mxVerifyVectorDimension(const mxArray *x, std::size_t n, const char *nx) {
  if (!( (mxGetM(x) == n && mxGetN(x) == 1)
      || (mxGetM(x) == 1 && mxGetN(x) == n) )) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be a %u dim vector.", nx, n);
  }
}

void mxVerifyMatrixDimensions(const mxArray *x, std::size_t m, std::size_t n,
                        const char *nx) {
  if (m > 0 && mxGetM(x) != m) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must have %u row(s).", nx, m);
  }
  if (n > 0 && mxGetN(x) != n) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must have %u column(s).", nx, n);
  }
}

void mxVerifyMatrixSquare(const mxArray *x, const char *nx) {
  if (mxGetM(x) != mxGetN(x)) {
    mexErrMsgIdAndTxt(errInvalidArgument, "%s must be a square matrix.", nx);
  }
}

}

#endif // SDCA_MEX_UTIL_HPP

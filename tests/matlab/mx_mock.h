// An in-memory implementation of the mx and mex calls of the gateways (no MATLAB
// needed), for tests/matlab/*_mock.cpp; include it once per test binary, after
// the gateway source. Arrays are 2-D: numeric (double, single, logical) with raw
// column-major data, sparse (ir, jc), char and struct; errors throw.
#pragma once
#include <mex.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct mxArray_tag {
  mwSize m = 1, n = 1;
  mxClassID type = mxDOUBLE_CLASS;
  bool sparse = false;
  std::vector<unsigned char> raw;
  std::vector<mwIndex> ir, jc;
  std::string text;
  // Struct fields: names and values (element-major) of a created struct, or the
  // fields map of a struct built by a test (one element).
  std::vector<std::string> names;
  std::vector<mxArray*> values;
  std::map<std::string, mxArray*> fields;
};

namespace mock {

// MATLAB's mexErrMsgIdAndTxt leaves without unwinding, so std::cout must be restored before.
inline std::streambuf*& original_cout() {
  static std::streambuf* buffer = nullptr;
  return buffer;
}
inline int& errors_with_cout_redirected() {
  static int count = 0;
  return count;
}

inline std::size_t element_size(const mxClassID type) {
  switch (type) {
    case mxDOUBLE_CLASS:
      return sizeof(double);
    case mxSINGLE_CLASS:
      return sizeof(float);
    case mxLOGICAL_CLASS:
      return sizeof(mxLogical);
    case mxINT32_CLASS:
      return 4;
    default:
      return 0;
  }
}

inline mxArray* numeric(const mwSize m, const mwSize n, const mxClassID type) {
  auto a = new mxArray;
  a->m = m;
  a->n = n;
  a->type = type;
  a->raw.assign(m * n * element_size(type), 0);
  return a;
}

// A double, single or logical (or int32) m-by-n array of the given values.
inline mxArray* array(const mwSize m, const mwSize n, const std::vector<double>& data,
                      const mxClassID type = mxDOUBLE_CLASS) {
  mxArray* a = numeric(m, n, type);
  for (std::size_t i = 0; i < data.size() && i < m * n; ++i) {
    if (type == mxDOUBLE_CLASS)
      reinterpret_cast<double*>(a->raw.data())[i] = data[i];
    else if (type == mxSINGLE_CLASS)
      reinterpret_cast<float*>(a->raw.data())[i] = static_cast<float>(data[i]);
    else if (type == mxLOGICAL_CLASS)
      reinterpret_cast<mxLogical*>(a->raw.data())[i] = data[i] != 0;
    else if (type == mxINT32_CLASS)
      reinterpret_cast<std::int32_t*>(a->raw.data())[i] = static_cast<std::int32_t>(data[i]);
  }
  return a;
}

inline mxArray* scalar(const double value) { return array(1, 1, {value}); }

inline mxArray* string(const std::string& text) {
  auto a = new mxArray;
  a->type = mxCHAR_CLASS;
  a->m = 1;
  a->n = text.size();
  a->text = text;
  return a;
}

// A sparse m-by-n array of the (row, column) entries, with spare capacity.
inline mxArray* sparse(const mwSize m, const mwSize n,
                       const std::vector<std::pair<mwIndex, mwIndex>>& entries,
                       const mxClassID type = mxDOUBLE_CLASS, const std::size_t spare = 4) {
  auto a = new mxArray;
  a->m = m;
  a->n = n;
  a->type = type;
  a->sparse = true;
  a->jc.assign(n + 1, 0);
  for (mwIndex j = 0; j < n; ++j) {
    for (const auto& e : entries)
      if (e.second == j) a->ir.push_back(e.first);
    a->jc[j + 1] = a->ir.size();
  }
  const std::size_t nnz = a->ir.size();
  a->ir.resize(nnz + spare, 0);
  a->raw.assign((nnz + spare) * element_size(type), 0);
  for (std::size_t i = 0; i < nnz; ++i) {
    if (type == mxLOGICAL_CLASS)
      reinterpret_cast<mxLogical*>(a->raw.data())[i] = true;
    else
      reinterpret_cast<double*>(a->raw.data())[i] = 1;
  }
  return a;
}

inline mxArray* options(const std::map<std::string, mxArray*>& fields) {
  auto a = new mxArray;
  a->type = mxSTRUCT_CLASS;
  a->fields = fields;
  return a;
}

template <typename T>
inline const T* data(const mxArray* a) {
  return reinterpret_cast<const T*>(a->raw.data());
}

}  // namespace mock

extern "C" {
void mexErrMsgIdAndTxt(const char*, const char* fmt, ...) {
  if (std::cout.rdbuf() != mock::original_cout()) ++mock::errors_with_cout_redirected();
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  throw std::runtime_error(buffer);
}
void mexErrMsgTxt(const char* text) { throw std::runtime_error(text); }
int mexPrintf(const char*, ...) { return 0; }
int mexEvalString(const char*) { return 0; }
void* mxCalloc(mwSize n, mwSize size) { return std::calloc(n, size); }
void mxFree(void* p) { std::free(p); }
mwSize mxGetM(const mxArray* a) { return a->m; }
mwSize mxGetN(const mxArray* a) { return a->n; }
mwSize mxGetNumberOfElements(const mxArray* a) { return a->m * a->n; }
mwSize mxGetNzmax(const mxArray* a) { return a->ir.size(); }
mwSize mxGetNumberOfDimensions(const mxArray*) { return 2; }
mxClassID mxGetClassID(const mxArray* a) { return a->type; }
bool mxIsDouble(const mxArray* a) { return a->type == mxDOUBLE_CLASS; }
bool mxIsSingle(const mxArray* a) { return a->type == mxSINGLE_CLASS; }
bool mxIsLogical(const mxArray* a) { return a->type == mxLOGICAL_CLASS; }
bool mxIsNumeric(const mxArray* a) {
  return mxIsDouble(a) || mxIsSingle(a) || a->type == mxINT32_CLASS;
}
bool mxIsStruct(const mxArray* a) { return a->type == mxSTRUCT_CLASS; }
bool mxIsCell(const mxArray*) { return false; }
bool mxIsChar(const mxArray* a) { return a->type == mxCHAR_CLASS; }
bool mxIsComplex(const mxArray*) { return false; }
bool mxIsSparse(const mxArray* a) { return a->sparse; }
bool mxIsEmpty(const mxArray* a) { return a->m * a->n == 0; }
bool mxIsScalar(const mxArray* a) { return a->m * a->n == 1; }
void* mxGetData(const mxArray* a) { return const_cast<unsigned char*>(a->raw.data()); }
double* mxGetPr(const mxArray* a) { return static_cast<double*>(mxGetData(a)); }
double mxGetScalar(const mxArray* a) {
  if (a->raw.empty()) return 0;
  switch (a->type) {
    case mxSINGLE_CLASS:
      return mock::data<float>(a)[0];
    case mxLOGICAL_CLASS:
      return mock::data<mxLogical>(a)[0];
    case mxINT32_CLASS:
      return mock::data<std::int32_t>(a)[0];
    default:
      return mock::data<double>(a)[0];
  }
}
mwIndex* mxGetIr(const mxArray* a) { return const_cast<mwIndex*>(a->ir.data()); }
mwIndex* mxGetJc(const mxArray* a) { return const_cast<mwIndex*>(a->jc.data()); }
mxArray* mxGetField(const mxArray* a, mwIndex i, const char* name) {
  if (!a->names.empty()) {
    for (std::size_t f = 0; f < a->names.size(); ++f)
      if (a->names[f] == name) return a->values[i * a->names.size() + f];
    return nullptr;
  }
  auto it = a->fields.find(name);
  return it == a->fields.end() ? nullptr : it->second;
}
int mxGetNumberOfFields(const mxArray* a) {
  return static_cast<int>(a->names.empty() ? a->fields.size() : a->names.size());
}
const char* mxGetFieldNameByNumber(const mxArray* a, int i) {
  if (!a->names.empty()) return a->names[static_cast<std::size_t>(i)].c_str();
  auto it = a->fields.begin();
  std::advance(it, i);
  return it->first.c_str();
}
int mxGetFieldNumber(const mxArray* a, const char* name) {
  for (int i = 0; i < mxGetNumberOfFields(a); ++i)
    if (std::string(mxGetFieldNameByNumber(a, i)) == name) return i;
  return -1;
}
int mxGetString(const mxArray* a, char* buffer, mwSize length) {
  if (!mxIsChar(a) || length == 0) return 1;
  std::strncpy(buffer, a->text.c_str(), length - 1);
  buffer[length - 1] = 0;
  return 0;
}
mxArray* mxDuplicateArray(const mxArray* a) { return new mxArray(*a); }
void mxDestroyArray(mxArray* a) { delete a; }
mxArray* mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID type, mxComplexity) {
  return mock::numeric(m, n, type);
}
mxArray* mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity c) {
  return mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, c);
}
mxArray* mxCreateDoubleScalar(double value) { return mock::scalar(value); }
mxArray* mxCreateLogicalScalar(mxLogical value) {
  return mock::array(1, 1, {value ? 1.0 : 0.0}, mxLOGICAL_CLASS);
}
mxArray* mxCreateString(const char* text) { return mock::string(text); }
mxArray* mxCreateStructMatrix(mwSize m, mwSize n, int count, const char** names) {
  auto a = new mxArray;
  a->type = mxSTRUCT_CLASS;
  a->m = m;
  a->n = n;
  for (int f = 0; f < count; ++f) a->names.push_back(names[f]);
  a->values.assign(m * n * static_cast<std::size_t>(count), nullptr);
  return a;
}
void mxSetFieldByNumber(mxArray* a, mwIndex i, int f, mxArray* value) {
  a->values[i * a->names.size() + static_cast<std::size_t>(f)] = value;
}
}

// Runs the MATLAB training gateway's setup helpers on an in-memory mxArray mock
// (no MATLAB needed): warm starts and sparse targets, the k domain of every objective, and
// nonfinite data.
// Built and run by tests/matlab/run_gateway_mock.sh.
#include <mex.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct mxArray_tag {
  mwSize m = 0, n = 0;
  mxClassID type = mxDOUBLE_CLASS;
  bool sparse = false;
  std::vector<double> data;
  std::vector<mwIndex> ir, jc;
  std::map<std::string, mxArray*> fields;
};

#include "mex_solve.cpp"

// MATLAB's mexErrMsgIdAndTxt leaves without unwinding, so std::cout must be restored before.
std::streambuf* original_cout = nullptr;
int errors_with_cout_redirected = 0;

extern "C" {
void mexErrMsgIdAndTxt(const char*, const char* fmt, ...) {
  if (std::cout.rdbuf() != original_cout) ++errors_with_cout_redirected;
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  throw std::runtime_error(buffer);
}
void mexErrMsgTxt(const char* text) { throw std::runtime_error(text); }
int mexPrintf(const char*, ...) { return 0; }
int mexEvalString(const char*) { return 0; }
mwSize mxGetM(const mxArray* a) { return a->m; }
mwSize mxGetN(const mxArray* a) { return a->n; }
mwSize mxGetNumberOfElements(const mxArray* a) { return a->m * a->n; }
mwSize mxGetNzmax(const mxArray* a) { return a->ir.size(); }
mwSize mxGetNumberOfDimensions(const mxArray*) { return 2; }
mxClassID mxGetClassID(const mxArray* a) { return a->type; }
bool mxIsDouble(const mxArray* a) { return a->type == mxDOUBLE_CLASS; }
bool mxIsSingle(const mxArray* a) { return a->type == mxSINGLE_CLASS; }
bool mxIsLogical(const mxArray* a) { return a->type == mxLOGICAL_CLASS; }
bool mxIsNumeric(const mxArray* a) { return mxIsDouble(a) || mxIsSingle(a); }
bool mxIsStruct(const mxArray* a) { return a->type == mxSTRUCT_CLASS; }
bool mxIsCell(const mxArray*) { return false; }
bool mxIsChar(const mxArray*) { return false; }
bool mxIsComplex(const mxArray*) { return false; }
bool mxIsSparse(const mxArray* a) { return a->sparse; }
bool mxIsEmpty(const mxArray* a) { return a->m * a->n == 0; }
bool mxIsScalar(const mxArray* a) { return a->m * a->n == 1; }
void* mxGetData(const mxArray* a) { return const_cast<double*>(a->data.data()); }
double* mxGetPr(const mxArray* a) { return static_cast<double*>(mxGetData(a)); }
double mxGetScalar(const mxArray* a) { return a->data.empty() ? 0 : a->data[0]; }
mwIndex* mxGetIr(const mxArray* a) { return const_cast<mwIndex*>(a->ir.data()); }
mwIndex* mxGetJc(const mxArray* a) { return const_cast<mwIndex*>(a->jc.data()); }
mxArray* mxGetField(const mxArray* a, mwIndex, const char* name) {
  auto it = a->fields.find(name);
  return it == a->fields.end() ? nullptr : it->second;
}
int mxGetFieldNumber(const mxArray* a, const char* name) {
  int i = 0;
  for (const auto& f : a->fields) {
    if (f.first == name) return i;
    ++i;
  }
  return -1;
}
mxArray* mxDuplicateArray(const mxArray* a) { return new mxArray(*a); }
void mxDestroyArray(mxArray* a) { delete a; }
mxArray* mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID type, mxComplexity) {
  auto a = new mxArray;
  a->m = m;
  a->n = n;
  a->type = type;
  a->data.resize(m * n);
  return a;
}
mxArray* mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity c) {
  return mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, c);
}
}

namespace {

int failures = 0;

void check(const bool ok, const std::string& what) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
  if (!ok) ++failures;
}

mxArray matrix(const mwSize m, const mwSize n, const std::vector<double>& data) {
  mxArray a;
  a.m = m;
  a.n = n;
  a.data = data;
  return a;
}

// Warm start: training from initial_dual (opts.A) starts at W = W0 + X A'; a given W is ignored.
void warm_starts() {
  double x[] = {1, 0, 0, 1, -1, -1};  // d = 2, n = 3, column-major
  mxArray a = matrix(3, 3, {.6, -.3, -.3, -.3, .6, -.3, -.3, -.3, .6});
  mxArray center = matrix(2, 3, {.5, .3, -1, .1, .2, -.4});
  mxArray bogus = matrix(2, 3, {99, 99, 99, 99, 99, 99});
  for (bool warm : {false, true})
    for (bool centered : {false, true})
      for (bool given_w : {false, true}) {
        auto in = sdca::make_input_feature(2, 3, x);
        std::vector<sdca::size_type> y = {0, 1, 2};
        sdca::multiclass_output out(3, y);
        mxArray opts;
        opts.type = mxSTRUCT_CLASS;
        if (warm) opts.fields["A"] = &a;
        if (centered) opts.fields["W0"] = &center;
        if (given_w) opts.fields["W"] = &bogus;
        sdca::model_info<mxArray*> info;
        double *A = nullptr, *W = nullptr, *W0 = nullptr;
        std::string what = "warm=" + std::to_string(warm) + " center=" + std::to_string(centered) +
                           " W=" + std::to_string(given_w);
        try {
          set_variables(&opts, in, out, info, A, W, W0);
        } catch (const std::exception& e) {
          check(false, "warm start " + what + ": " + e.what());
          continue;
        }
        double error = 0;
        for (int j = 0; j < 3; ++j)
          for (int l = 0; l < 2; ++l) {
            double expected = centered ? center.data[l + 2 * j] : 0;
            for (int i = 0; i < 3; ++i) expected += x[l + 2 * i] * (warm ? a.data[j + 3 * i] : 0);
            error = std::max(error, std::abs(W[l + 2 * j] - expected));
          }
        check(error <= 1e-15, "warm start W = W0 + X A' " + what);
      }
}

// Sparse target: a 4-class sparse target with spare capacity (nzmax 6 > nnz 2) and classes
// {1, 2} keeps 4 classes and zero-based labels [1, 2], double or logical.
void sparse_targets() {
  for (mxClassID type : {mxDOUBLE_CLASS, mxLOGICAL_CLASS}) {
    mxArray labels;
    labels.m = 4;
    labels.n = 2;
    labels.sparse = true;
    labels.type = type;
    labels.data = {1, 1, 0, 0, 0, 0};
    labels.ir = {1, 2, 0, 0, 0, 0};
    labels.jc = {0, 1, 2};
    const std::string what = type == mxLOGICAL_CLASS ? "logical" : "double";
    try {
      validate_labels(&labels, 2);
      auto out = make_output<sdca::multilabel_output>(&labels, 2);
      check(out.num_classes == 4 && out.labels == std::vector<sdca::size_type>({1, 2}),
            "sparse " + what + " keeps classes 4 and labels 1, 2");
    } catch (const std::exception& e) {
      check(false, "sparse " + what + ": " + e.what());
    }
  }
}

// matsdca_fit rejects k outside [1, classes - 1] for every objective, also the multilabel
// objectives that do not use k, before it trains.
void k_domain() {
  double x[] = {1, 0, 0, 1, -1, -1};  // d = 2, n = 3, column-major
  for (double k : {0.0, 3.0, 10.0}) {
    mxArray value = matrix(1, 1, {k});
    mxArray opts;
    opts.type = mxSTRUCT_CLASS;
    opts.fields["k"] = &value;
    const std::string what = " k=" + std::to_string(static_cast<int>(k));
    for (const char* obj : {"cross_entropy", "softmax", "l2_entropy_topk", "multiclass_hinge",
                            "topk_hinge_alpha", "topk_hinge_beta"}) {
      std::vector<sdca::size_type> y = {0, 1, 2};
      try {
        set_objective<double, double>(nullptr, nullptr, &opts, obj,
                                      sdca::make_input_feature(2, 3, x),
                                      sdca::multiclass_output(3, y));
        check(false, std::string("k ") + obj + what + " accepted");
      } catch (const std::exception& e) {
        check(std::string(e.what()).find("'k'") != std::string::npos,
              std::string("k ") + obj + what + ": " + e.what());
      }
    }
    for (const char* obj : {"multilabel_hinge", "mlsoftmax", "mlsvm", "mlsvm_smooth"}) {
      std::vector<sdca::size_type> labels = {0, 1, 2}, offsets = {0, 1, 2, 3};
      try {
        set_objective<double, double>(nullptr, nullptr, &opts, obj,
                                      sdca::make_input_feature(2, 3, x),
                                      sdca::multilabel_output(3, labels, offsets));
        check(false, std::string("k ") + obj + what + " accepted");
      } catch (const std::exception& e) {
        check(std::string(e.what()).find("'k'") != std::string::npos,
              std::string("k ") + obj + what + ": " + e.what());
      }
    }
  }
}

// matsdca_fit rejects nonfinite training data before it trains.
void nonfinite_data() {
  for (double bad : {NAN, INFINITY}) {
    mxArray data = matrix(2, 3, {1, 0, 0, bad, -1, -1});
    mxArray labels = matrix(1, 3, {1, 2, 3});
    mxArray opts;
    opts.type = mxSTRUCT_CLASS;
    const mxArray* prhs[] = {&data, &labels};
    mxArray* plhs[2] = {nullptr, nullptr};
    const std::string what = std::isnan(bad) ? "NaN" : "inf";
    try {
      set_input<double, double>(plhs, prhs, &opts);
      check(false, "data with " + what + " accepted");
    } catch (const std::exception& e) {
      check(std::string(e.what()) == "data must be finite", "data with " + what + ": " + e.what());
    }
  }
}

// matsdca_fit restores std::cout (buffer and format) before it raises an error, for an
// unsupported data type and for an exception from training.
void cout_restored_on_error() {
  mxArray ints = matrix(2, 3, {1, 0, 0, 1, -1, -1});
  ints.type = mxINT32_CLASS;
  mxArray nonfinite = matrix(2, 3, {1, 0, 0, NAN, -1, -1});
  mxArray labels = matrix(1, 3, {1, 2, 3});
  mxArray opts;
  opts.type = mxSTRUCT_CLASS;
  for (const mxArray* data : {&ints, &nonfinite}) {
    const std::string what = data == &ints ? "int32 data" : "nonfinite data";
    const std::ios::fmtflags flags = std::cout.flags();
    const std::streamsize precision = std::cout.precision();
    const int before = errors_with_cout_redirected;
    const mxArray* prhs[] = {data, &labels, &opts};
    mxArray* plhs[1] = {nullptr};
    bool raised = false;
    try {
      mexFunction(1, plhs, 3, prhs);
    } catch (const std::exception&) {
      raised = true;
    }
    check(raised && errors_with_cout_redirected == before && std::cout.rdbuf() == original_cout &&
              std::cout.flags() == flags && std::cout.precision() == precision,
          "std::cout restored before the " + what + " error");
  }
}

}  // namespace

int main() {
  original_cout = std::cout.rdbuf();
  warm_starts();
  sparse_targets();
  k_domain();
  nonfinite_data();
  cout_restored_on_error();
  return failures == 0 ? 0 : 1;
}

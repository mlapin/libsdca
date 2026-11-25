#ifndef SDCA_CORE_EXECUTION_H
#define SDCA_CORE_EXECUTION_H

// The settings and results of a batch of rows (the same types for every instruction set).
#include <cstddef>

namespace sdca {
namespace kernel {

// The values of enum sdca_precision and enum sdca_result (sdca/c_api.h).
enum Precision { precision_default = 0, precision_float32 = 1, precision_float64 = 2 };
enum Status { status_ok = 0, status_invalid_argument = -2, status_unsupported = -3 };

struct Execution {
  int precision;            // Precision
  std::size_t num_threads;  // 0: SDCA_NUM_THREADS, else automatic; n: up to n
  bool check_rows;          // every row checked before any output is written
};

// status_ok with the count of invalid rows (NaN outputs) and the first of them (n if none);
// with check_rows, status_invalid_argument where a row is invalid, nothing written.
// status_invalid_argument also for float32 work on double storage; status_unsupported for a
// loss kind without kernels or targets the loss does not take.
struct Result {
  int status;
  std::size_t invalid_rows;
  std::size_t first_invalid;
};

}  // namespace kernel
}  // namespace sdca

#endif  // SDCA_CORE_EXECUTION_H

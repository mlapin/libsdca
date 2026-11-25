// The CBLAS functions of libsdca: a BLAS is external to the builds (src/native_build.txt) when
// this program links against it.
#include "sdca/math/cblas.h"

int main() {
  float x = 1;
  double y = 1;
  cblas_sscal(1, 1, &x, 1);
  cblas_dscal(1, 1, &y, 1);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1, &x, 1, &x, 1, 0, &x, 1);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1, &y, 1, &y, 1, 0, &y, 1);
  return cblas_sdot(1, &x, 1, &x, 1) == cblas_ddot(1, &y, 1, &y, 1) ? 0 : 1;
}

#!/usr/bin/env bash
# Compile the mex sources against a stub mex.h (declarations only) to check
# that they still build against the headers; no MATLAB installation needed.
set -euo pipefail
cd "$(dirname "$0")/../.."
for target in solve prox loss; do
  g++ -std=c++11 -O1 -c -Wall -Wextra -Wshadow -Wdouble-promotion \
    -DBLAS_DEFAULT_LOCAL_HEADER -DMEX_SOLVE='"libsdca_solve"' \
    -DMEX_PROX='"libsdca_prox"' -DMEX_LOSS='"libsdca_loss"' -DLIBSDCA_VERSION='"check"' \
    -Iinclude -Itests/matlab -Isrc/matlab "src/matlab/mex_$target.cpp" -o /dev/null
  echo "mex_$target.cpp: ok"
done

#!/usr/bin/env bash
# Build the mex files with GNU Octave and run the Matlab smoke tests:
#   OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh
# Octave's mex API covers everything the gateways use; the CBLAS library of
# the Octave installation (or SDCA_BLAS) is linked. No Matlab is needed.
set -euo pipefail
cd "$(dirname "$0")/../.."
prefix=${OCTAVE_HOME:-$(dirname "$(dirname "$(command -v octave-cli)")")}
mkoctfile=$prefix/bin/mkoctfile
if [[ -n "${SDCA_BLAS:-}" ]]; then
  blas="$SDCA_BLAS"
  blas_def="-DBLAS_DEFAULT_LOCAL_HEADER"
elif [[ -f "$prefix/lib/libcblas.so" || -f "/usr/lib/x86_64-linux-gnu/libcblas.so" ]]; then
  blas="-L$prefix/lib -lcblas -Wl,-rpath,$prefix/lib"
  blas_def="-DBLAS_DEFAULT_LOCAL_HEADER"
else
  blas=""
  blas_def="-DBLAS_BUILTIN"
fi
out=${1:-build/octave}; mkdir -p "$out"; out=$(cd "$out" && pwd)
# mkoctfile strips quotes from -D values; string macros go through a header.
printf '#define SDCA_VERSION "octave"\n#define MEX_PROX "matsdca_prox"\n#define MEX_SOLVE "matsdca_fit"\n#define MEX_LOSS "matsdca_loss"\n' > "$out/defines.h"
# The numerics flags of matlab/make.m and the CMake build.
flags=(--mex "$blas_def" -include "$out/defines.h" -Iinclude -Isrc/matlab)
# Some relocatable Octave packages pad rewritten compiler-flag strings with NULs.
# Passing those through mkoctfile's shell command truncates it before the source file.
# Preserve the configured flags, stripping only that padding before overriding them.
for variable in CPPFLAGS CFLAGS LDFLAGS DL_LDFLAGS; do
  if [[ -z "${!variable+x}" ]]; then
    value=$("$mkoctfile" -p "$variable" | tr -d '\000')
    export "$variable=$value"
  fi
done
export OCTAVE_HOME=$prefix
# Prefer the compiler shipped with this Octave prefix when its configured name is
# relative; a user's explicit CC/CXX still wins (for example Apple versus LLVM clang).
for variable in CC CXX; do
  if [[ -z "${!variable:-}" ]]; then
    value=$("$mkoctfile" -p "$variable")
    [[ ! -x "$prefix/bin/$value" ]] || value="$prefix/bin/$value"
    export "$variable=$value"
  fi
done
export DL_LD=$CXX LD_CXX=$CXX CXXFLAGS="-std=c++11 -O2 -fno-fast-math -ffp-contract=off -fPIC -pthread"
"$mkoctfile" "${flags[@]}" -o "$out/matsdca_prox.mex" src/matlab/mex_prox.cpp $blas
"$mkoctfile" "${flags[@]}" -o "$out/matsdca_fit.mex" src/matlab/mex_solve.cpp $blas
"$mkoctfile" "${flags[@]}" -o "$out/matsdca_loss.mex" src/matlab/mex_loss.cpp $blas
"$prefix/bin/octave-cli" --no-gui --eval "page_screen_output(false); addpath('$out'); addpath('matlab'); addpath('tests/matlab'); test_numerics; disp('octave: ok')"
"$prefix/bin/octave-cli" --no-gui --eval "page_screen_output(false); addpath('$out'); addpath('matlab'); demo; disp('octave demo: ok')"

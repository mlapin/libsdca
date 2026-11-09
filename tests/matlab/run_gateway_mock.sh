#!/usr/bin/env bash
# Build tests/matlab/gateway_mock.cpp against the stub mex.h and run it; no MATLAB
# needed. The gateway's other mx calls are never made, so they stay unresolved.
# Optional argument: another source tree whose include/ and src/matlab/ to test.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tree="${1:-$here/../..}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
# The linker flag that leaves the unused mx symbols unresolved: Apple's ld has no GNU option.
if [ "$(uname -s)" = Darwin ]; then
  unresolved=-Wl,-undefined,dynamic_lookup
else
  unresolved=-Wl,--unresolved-symbols=ignore-all
fi
g++ -std=c++11 -O1 -DBLAS_BUILTIN -ffp-contract=off \
  -DMEX_SOLVE='"libsdca_solve"' -DLIBSDCA_VERSION='"check"' \
  -I"$tree/include" -I"$here" -I"$tree/src/matlab" "$here/gateway_mock.cpp" \
  "$unresolved" -o "$out/gateway_mock"
"$out/gateway_mock"

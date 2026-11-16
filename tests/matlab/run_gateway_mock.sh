#!/usr/bin/env bash
# Build tests/matlab/gateway_mock.cpp (matsdca_fit) and loss_mock.cpp (matsdca_loss)
# against the stub mex.h and the in-memory mx implementation of mx_mock.h, and run
# them; no MATLAB needed. Any mx call the mock lacks stays unresolved.
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
for test in gateway_mock loss_mock; do
  g++ -std=c++11 -O1 -pthread -DBLAS_BUILTIN -ffp-contract=off \
    -DMEX_SOLVE='"libsdca_solve"' -DMEX_LOSS='"libsdca_loss"' -DLIBSDCA_VERSION='"check"' \
    -I"$tree/include" -I"$here" -I"$tree/src/matlab" "$here/$test.cpp" \
    "$unresolved" -o "$out/$test"
  "$out/$test"
done

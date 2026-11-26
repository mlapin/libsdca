#!/usr/bin/env bash
# Vendor the C++ sources into rust/native and build the crate archive, which
# builds without the repository: rust/package.sh [cargo package arguments].
set -euo pipefail
crate=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$crate")
rm -rf "$crate/native"
mkdir -p "$crate/native/src"
cp -R "$root/include" "$crate/native/include"
# The C ABI and the build settings and BLAS probe of src/native_build.txt.
cp "$root/src/c_api.cpp" "$root/src/native_build.txt" "$root/src/blas_probe.cpp" "$crate/native/src/"
cp "$root/LICENSE" "$crate/LICENSE"
cd "$crate"
# The vendored copies are untracked by design.
cargo package --allow-dirty "$@"

#!/usr/bin/env bash
# Vendor the C++ sources into rust/native and build the crate archive, which
# builds without the repository: rust/package.sh [cargo package arguments].
set -euo pipefail
crate=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$crate")
rm -rf "$crate/native"
mkdir -p "$crate/native/src"
cp -R "$root/include" "$crate/native/include"
cp "$root/src/c_api.cpp" "$root/src/tensor_losses.h" "$crate/native/src/"
cp "$root/LICENSE" "$crate/LICENSE"
cd "$crate"
# The vendored copies are untracked by design.
cargo package --allow-dirty "$@"

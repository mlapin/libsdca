#!/usr/bin/env bash
# Link this crate against one apt BLAS package in one SDCA_BLAS form and run it:
# ci.sh builtin, or ci.sh reference|openblas|blis l|l-file|abs-so|static|L-l
set -euo pipefail
here=$(dirname "$0")
if [[ $1 == builtin ]]; then
  unset SDCA_BLAS
  exec bash "$here/check.sh" builtin
fi
case $1 in
  reference) pkg=libblas-dev name=blas ;;
  openblas) pkg=libopenblas-pthread-dev name=openblas ;;
  blis) pkg=libblis-pthread-dev name=blis ;;
  *) echo "unknown provider $1" >&2; exit 2 ;;
esac
so=$(dpkg -L "$pkg" | grep "/lib$name\.so$" | head -1)
dir=$(dirname "$so")
soname=$(readelf -d "$so" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')
mode=shared
case $2 in
  l) SDCA_BLAS=-l$name ;;
  l-file) SDCA_BLAS=-l:$soname ;;
  abs-so) SDCA_BLAS=$so ;;
  static) SDCA_BLAS=$dir/lib$name.a mode=static ;;
  L-l) SDCA_BLAS="-L$dir -l$name" ;;
  *) echo "unknown form $2" >&2; exit 2 ;;
esac
# Debian's static OpenBLAS runs its initializer from .ctors, which GNU ld moves
# into .init_array and lld (Rust's default linker since 1.90) does not.
if [[ $1/$2 == openblas/static ]]; then export RUSTFLAGS="-C link-arg=-fuse-ld=bfd"; fi
export SDCA_BLAS
exec bash "$here/check.sh" "$mode" "$soname"

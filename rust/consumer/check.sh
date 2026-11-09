#!/usr/bin/env bash
# Build and run this downstream crate, linked as SDCA_BLAS says (unset: the
# built-in BLAS), and check the link: check.sh builtin | shared <soname> | static
set -euo pipefail
cd "$(dirname "$0")"
mode=$1
echo "SDCA_BLAS=${SDCA_BLAS:-} ($mode)"
cargo build --quiet
bin=${CARGO_TARGET_DIR:-target}/debug/sdca-consumer
needed=$(readelf -d "$bin" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
# The linker keeps only the CBLAS functions the crate calls (no gemm in this one).
cblas=$(nm "$bin" | grep -c ' T cblas_' || true)
blas_so=$(grep -E '^lib(c?blas|openblas|blis|mkl)' <<<"$needed" || true)
case $mode in
  builtin) [[ -z $blas_so && $cblas == 0 ]] ;;
  shared) [[ $blas_so == "$2" && $cblas == 0 ]] ;;
  static) [[ -z $blas_so && $cblas -gt 0 ]] ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac || { echo "unexpected link: NEEDED ${blas_so:-none}, CBLAS functions defined: $cblas" >&2; exit 1; }
echo "linked: ${blas_so:-no shared BLAS}, CBLAS functions defined in the binary: $cblas"
"$bin"

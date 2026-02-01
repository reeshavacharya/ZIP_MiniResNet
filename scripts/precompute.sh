#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo
python3 -W ignore ../src/piecewise_polynomial_approximation/approx.py relu
python3 ../src/piecewise_polynomial_approximation/extract.py relu
rm -f -- "../src/piecewise_polynomial_approximation/relu_approx.py"


src="../src/piecewise_polynomial_approximation/precomputed_lookup_tables_ieee754_hex"
dst="../src/proof_generation/ZIP_proof_generation"

mkdir -p -- "$dst"
rm -rf -- "$dst/precomputed_lookup_tables_ieee754_hex"
mv -- "$src" "$dst/"

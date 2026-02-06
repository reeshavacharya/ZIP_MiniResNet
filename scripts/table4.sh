#!/usr/bin/env bash
set -euo pipefail

# ---------------- Args ----------------
# 0 (default) => normal/verbose output
# 1           => quiet mode (only print the two totals at the end)
VERBOSE_FLAG="${1:-0}"
VALUES_DIR="${2:-y_yprime_examples}"

if [[ ! "$VERBOSE_FLAG" =~ ^[01]$ ]]; then
  echo "Usage: $0 [0|1] [values_dir] [\"activations list\"]" >&2
  echo "  0 = verbose (default), 1 = quiet totals only" >&2
  echo "  values_dir defaults to 'y_yprime_examples'" >&2
  echo "  activations defaults to 'gelu' (e.g. \"gelu elu selu softmax layernorm\")" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
TIMES_FILE="$REPO_ROOT/proof_times.txt"

# ---- reset previous run's totals file ----
rm -f -- "$TIMES_FILE"

# ---------------- Paths ----------------
EXAMPLES_ROOT="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup/examples"
Y_YPRIME_VALUES_DIR_PATH="$EXAMPLES_ROOT/$VALUES_DIR"
Y_YPRIME_VALUES_DIR_NAME="$VALUES_DIR"

if [[ ! -d "$Y_YPRIME_VALUES_DIR_PATH" ]]; then
  echo "ERROR: examples directory not found: $Y_YPRIME_VALUES_DIR_PATH" >&2
  echo "Hint: pass it as the 2nd arg, e.g. './scripts/table4.sh 0 y_yprime_examples'" >&2
  exit 1
fi

if [[ "$VERBOSE_FLAG" == "1" ]]; then
  exec 3>&1 4>&2
  exec >/dev/null 2>&1
fi

CIRCUIT_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/non-linear"
CAULK_DIR="$REPO_ROOT/src/proof_generation/caulk"
ZIP_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup"
LOOKUP_TABLE_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/precomputed_lookup_tables_ieee754_hex"
TARGET_DIR="$LOOKUP_TABLE_DIR/target_table"

mkdir -p "$CAULK_DIR/polys"

cleanup() { rm -rf -- "$TARGET_DIR"; }
trap cleanup EXIT

# ---------------- Preset values (GELU) ----------------
declare -A GELU_PRESET=(
  [TABLE_SIZE]=70
  [PRIVATE_VECTOR_SIZE]=10
  [TABLE_PRIME_SIZE]=8
  [PRIVATE_VECTOR_PRIME_SIZE]=2
)

echo "Running table4 evaluations..."
pushd "$CIRCUIT_DIR" >/dev/null

python3 "$ZIP_DIR/add_interval_index.py" \
    --pairs "$Y_YPRIME_VALUES_DIR_PATH/gelu_y_yprime.txt" \
    --intervals "$LOOKUP_TABLE_DIR/gelu_intervals_ieee754.txt"
echo
  echo "Starting evaluations for activation: gelu"

pairs_file="$Y_YPRIME_VALUES_DIR_PATH/gelu_y_yprime.txt"
if [[ ! -f "$pairs_file" ]]; then
  echo "ERROR: Y/Y' pairs file not found: $pairs_file" >&2
  exit 1
fi

readarray -t pair_lines < <(
    grep -E '^[[:space:]]*(0[xX][0-9A-Fa-f]+|[0-9]+)[[:space:]]*,[[:space:]]*(0[xX][0-9A-Fa-f]+|[0-9]+)([[:space:]]*,[[:space:]]*(0[xX][0-9A-Fa-f]+|[0-9]+))?[[:space:]]*$' \
      "$pairs_file"
  )
instances="${#pair_lines[@]}"

# Optional cap on the number of instances to prove, to avoid
# running an extremely large gnark circuit that may exhaust
# memory. Set MAX_INSTANCES in the environment (e.g.
#   MAX_INSTANCES=512 ./scripts/table4.sh ... )
# If unset or non-positive, all instances are used.
if [[ -n "${MAX_INSTANCES:-}" ]] && [[ "$MAX_INSTANCES" =~ ^[0-9]+$ ]] && [[ "$MAX_INSTANCES" -gt 0 ]] && [[ "$instances" -gt "$MAX_INSTANCES" ]]; then
  instances="$MAX_INSTANCES"
  pair_lines=("${pair_lines[@]:0:instances}")
fi

if [[ "$instances" -lt 1 ]]; then
    echo "ERROR: No valid Y/Y' lines found in $pairs_file" >&2
    exit 1
fi
preset_name="GELU_PRESET"
declare -n PRESET="$preset_name"

python3 generate_config.py \
    --preset gelu \
    --instances "$instances" \
    --mode table \
  --proving "${ZIP_PROVING:-true}" \
  --values_dir "$Y_YPRIME_VALUES_DIR_NAME"

go run main.go config.go

# -------- Part 2 (Piecewise Polynomial Coefficients Lookup) --------
n_coeff=$(awk -v t="${PRESET[TABLE_SIZE]}" 'BEGIN{n=0;p=1;while(p<t){p*=2;n++}print n}')
n_prime=$(awk -v t="${PRESET[TABLE_PRIME_SIZE]}" 'BEGIN{n=0;p=1;while(p<t){p*=2;n++}print n}')

for ((idx=0; idx<instances; idx++)); do
  line="${pair_lines[$idx]}"
  IFS=',' read -r _y _yp _third <<<"$line"
  third_raw="$(echo "${_third:-}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"

  if [[ -z "$third_raw" ]]; then
    echo "ERROR: line $((idx+1)) has no 3rd field (block index)"; exit 1
  fi
  if [[ "$third_raw" =~ ^0[xX][0-9A-Fa-f]+$ ]]; then
    third_dec=$(( third_raw ))
  else
    third_dec=$(( 10#$third_raw ))
  fi

  interval_positions="${third_dec},$((third_dec + 1))"
  s=${PRESET[PRIVATE_VECTOR_SIZE]}
  start=$(( third_dec * s ))
  end=$(( start + s - 1 ))
  coeff_positions="$(seq -s, "$start" "$end")"

  run_i=$((idx+1))

  echo
  echo "gelu Coefficients lookup run ${run_i}/${instances}"
  mkdir -p -- "$TARGET_DIR"
  cp "$LOOKUP_TABLE_DIR/gelu_coefficients_ieee754.txt" \
    "$TARGET_DIR/target_lookup_table.txt"

  pushd "$ZIP_DIR" >/dev/null
  cargo run --release --example multi_lookup -- \
    --n "$n_coeff" \
    --m "${PRESET[PRIVATE_VECTOR_SIZE]}" \
    --positions "$coeff_positions" \
    --runs 1
  popd >/dev/null
  rm -rf -- "$TARGET_DIR"

  echo
  echo "gelu Intervals lookup run ${run_i}/${instances}"
  mkdir -p -- "$TARGET_DIR"
  cp "$LOOKUP_TABLE_DIR/gelu_intervals_ieee754.txt" \
    "$TARGET_DIR/target_lookup_table.txt"

  pushd "$ZIP_DIR" >/dev/null
  cargo run --release --example multi_lookup -- \
    --n "$n_prime" \
    --m "${PRESET[PRIVATE_VECTOR_PRIME_SIZE]}" \
    --positions "$interval_positions" \
    --runs 1
  popd >/dev/null
  rm -rf -- "$TARGET_DIR"
done

popd >/dev/null


if [[ "$VERBOSE_FLAG" == "1" ]]; then
  exec 1>&3 2>&4
fi

echo
echo "*********************************************"
# ---- Totals from proof_times.txt ----
if [[ -s "$TIMES_FILE" ]]; then
  awk -F',' '
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); p += $1 + 0; v += $2 + 0 }
    END {
      printf("total proving time : %.6f sec\n", p + 0);
      printf("total verification time: %.6f sec\n", v + 0);
    }' "$TIMES_FILE"
else
  echo "No proof times found at $TIMES_FILE"
fi

echo "*********************************************"
echo

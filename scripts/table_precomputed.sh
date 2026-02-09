#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Work in MiniResNet_DIR 
MiniResNet_DIR="$REPO_ROOT/src/mini-ResNet"
LINEAR_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear"
TIMES_FILE="$LINEAR_DIR/proof_times.txt" 
rm -f "$TIMES_FILE"
DEST="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear/mini_resnet_output_add_mult"

pushd "$MiniResNet_DIR" >/dev/null

echo
echo "Performance of ZIP on Mini_ResNet (precomputed model)"
echo

echo "=============================================================="
echo "== Proving non-linear layers (activations): ReLU =="
echo "=============================================================="

# NOTE: We assume mini_resnet_model_params.json already exists from prior training.
# We only run inference and proof generation here.
python infer_mini_resnet.py
python act_ops.py --key act1_input
python act_ops.py --key res1_act1_input
python act_ops.py --key res1_act2_input
python act_ops.py --key res2_act1_input
python act_ops.py --key res2_act2_input
python act_ops.py --key act2_input

# Cap the number of activation instances proved per call to table4.sh
# to avoid extremely large gnark circuits exhausting memory.
: "${MAX_INSTANCES:=4}"

echo "Proving 1st activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_act1 "relu"

echo "Proving 1st residual connections' 1st activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_res1_act1 "relu"

echo "Proving 1st residual connections' 2nd activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_res1_act2 "relu"

echo "Proving 2nd residual connections' 1st activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_res2_act1 "relu"

echo "Proving 2nd residual connections' 2nd activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_res2_act2 "relu"

echo "Proving 2nd activation (MAX_INSTANCES=$MAX_INSTANCES)"
"$SCRIPT_DIR/table4.sh" 0 y_yprime_examples_mini_resnet_act_act2 "relu"

echo ""
echo "==========================="
echo "== Proving linear layers =="
echo "==========================="

# Linear layers
rm -rf mini_resnet_output
python conv1_ops.py
python pool1_ops.py
python res1_conv1_ops.py
python res1_conv2_ops.py
python res2_conv1_ops.py
python res2_conv2_ops.py
python pool2_ops.py
python fc1_ops.py
python fc2_ops.py

CHUNK=60000

rm -rf "$DEST"
mkdir -p "$DEST/addition" "$DEST/multiplication"

pushd "$MiniResNet_DIR/mini_resnet_output" >/dev/null
  
split -d -a 5 -l "$CHUNK" --additional-suffix=.txt addition.txt       "$DEST/addition/addition_"
split -d -a 5 -l "$CHUNK" --additional-suffix=.txt multiplication.txt "$DEST/multiplication/multiplication_"

popd >/dev/null

pushd "$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear" >/dev/null

for i in {1..6}; do
  python generate_config.py --num-add 60000 --num-mul 60000 --size 18
  go run main.go config.go
done

python generate_config.py --num-add 60000 --num-mul 58096 --size 18
go run main.go config.go

python generate_config.py --num-add 9342 --num-mul 0 --size 16
go run main.go config.go

popd >/dev/null

echo
echo "*********************************************"
# ---- Totals from proof_times.txt ----
if [[ -s "$TIMES_FILE" ]]; then
  awk -F',' '
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); p += $1 + 0; v += $2 + 0 }
    END {
      printf("total proving time for all linear layers : %.6f sec\n", p + 0);
      printf("total verification time for all linear layers: %.6f sec\n", v + 0);
    }' "$TIMES_FILE"
else
  echo "No proof times found at $TIMES_FILE"
fi

echo "*********************************************"
echo

popd >/dev/null

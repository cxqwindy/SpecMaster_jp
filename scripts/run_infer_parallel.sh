#!/usr/bin/env bash
# Run standalone_infer_matching.py across N_GPUS GPUs in parallel.
# Each GPU gets an equal slice of the test file. Split stamps a global harness
# pair id on the shard copies; merge restores that order from the manifest.
#
# Usage:
#   bash scripts/run_infer_parallel.sh \
#       --model_path <ckpt> \
#       --test_file  <path/to/test.json> \
#       --output     <path/to/merged_result.json> \
#       [--n_gpus 8] [--max_turns 6]

set -euo pipefail

REPO=/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL
SPLIT_SCRIPT=/fs_mol/cxq/SpecMaster_jp/scripts/_split_and_merge_infer.py

MODEL_PATH=""
TEST_FILE=""
OUTPUT=""
N_GPUS=8
MAX_TURNS=6
USE_ROUTER=""
ROUTER_SIM_THRESHOLD=""
ROUTER_SIM_THRESHOLD_BOTH=""
ROUTER_SIM_THRESHOLD_STRICT=""
ROUTER_TEMPLATE_VARIANT_SEED=""
ROUTER_SKIP_MODEL_EVIDENCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)                MODEL_PATH="$2"; shift 2 ;;
    --test_file)                 TEST_FILE="$2";  shift 2 ;;
    --output)                    OUTPUT="$2";     shift 2 ;;
    --n_gpus)                    N_GPUS="$2";    shift 2 ;;
    --max_turns)                 MAX_TURNS="$2"; shift 2 ;;
    --use_router)                USE_ROUTER="1"; shift 1 ;;
    --router_sim_threshold)      ROUTER_SIM_THRESHOLD="$2"; shift 2 ;;
    --router_sim_threshold_both)   ROUTER_SIM_THRESHOLD_BOTH="$2"; shift 2 ;;
    --router_sim_threshold_strict) ROUTER_SIM_THRESHOLD_STRICT="$2"; shift 2 ;;
    --router_template_variant_seed) ROUTER_TEMPLATE_VARIANT_SEED="$2"; shift 2 ;;
    --router_skip_model_evidence) ROUTER_SKIP_MODEL_EVIDENCE="1"; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$MODEL_PATH" || -z "$TEST_FILE" || -z "$OUTPUT" ]] && \
  { echo "Usage: $0 --model_path <p> --test_file <f> --output <o> [--n_gpus 8]"; exit 1; }

TMPDIR=$(mktemp -d /tmp/infer_parallel_XXXXXX)
echo "[parallel-infer] tmpdir: $TMPDIR"
echo "[parallel-infer] splitting $TEST_FILE into $N_GPUS shards..."

python "$SPLIT_SCRIPT" split \
  --input "$TEST_FILE" \
  --n "$N_GPUS" \
  --outdir "$TMPDIR"

PIDS=()
for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
  SHARD="$TMPDIR/shard_${GPU_ID}.json"
  SHARD_OUT="$TMPDIR/result_${GPU_ID}.json"
  N_ROWS=$(python -c "import json; print(len(json.load(open('$SHARD'))))")
  echo "[parallel-infer] GPU $GPU_ID: $N_ROWS rows → $SHARD_OUT"
  CUDA_VISIBLE_DEVICES=$GPU_ID python "$REPO/recipe/nmr_matching_agent/standalone_infer_matching.py" \
    --model_path "$MODEL_PATH" \
    --from_test  "$SHARD" \
    --n          "$N_ROWS" \
    --output     "$SHARD_OUT" \
    --max_turns  "$MAX_TURNS" \
    ${USE_ROUTER:+--use_router} \
    ${ROUTER_SIM_THRESHOLD:+--router_sim_threshold "$ROUTER_SIM_THRESHOLD"} \
    ${ROUTER_SIM_THRESHOLD_BOTH:+--router_sim_threshold_both "$ROUTER_SIM_THRESHOLD_BOTH"} \
    ${ROUTER_SIM_THRESHOLD_STRICT:+--router_sim_threshold_strict "$ROUTER_SIM_THRESHOLD_STRICT"} \
    ${ROUTER_TEMPLATE_VARIANT_SEED:+--router_template_variant_seed "$ROUTER_TEMPLATE_VARIANT_SEED"} \
    ${ROUTER_SKIP_MODEL_EVIDENCE:+--router_skip_model_evidence} \
    > "$TMPDIR/log_${GPU_ID}.txt" 2>&1 &
  PIDS+=($!)
done

echo "[parallel-infer] waiting for ${#PIDS[@]} workers (pids: ${PIDS[*]})..."
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[parallel-infer] GPU $i done"
  else
    echo "[parallel-infer] GPU $i FAILED — see $TMPDIR/log_${i}.txt"
    FAILED=$((FAILED + 1))
  fi
done

if [[ $FAILED -gt 0 ]]; then
  echo "[parallel-infer] $FAILED worker(s) failed. Aborting merge."
  exit 1
fi

echo "[parallel-infer] merging results..."
SHARD_OUTS=$(for i in $(seq 0 $((N_GPUS - 1))); do echo "$TMPDIR/result_${i}.json"; done | tr '\n' ' ')
python "$SPLIT_SCRIPT" merge \
  --inputs $SHARD_OUTS \
  --manifest "$TMPDIR/shard_manifest.json" \
  --output "$OUTPUT"

echo "[parallel-infer] done → $OUTPUT"

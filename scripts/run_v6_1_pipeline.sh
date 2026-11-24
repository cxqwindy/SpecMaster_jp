#!/usr/bin/env bash
# run_v6_1_pipeline.sh  — Steps 2-6: rebuild all training data and launch v6.1 SFT
# Run from: /fs_mol/cxq/SpecMaster_jp
# Prerequisite: rebuild_prediction_cache.py must have completed (new cache is live).

set -euo pipefail

SPECMASTER=/fs_mol/cxq/SpecMaster_jp
RECIPES=/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent
LLF=/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory

echo "===== STEP 2a: build v6.1 pool (v5 replay + v6 hard negs + upsampled positives) ====="
python ${SPECMASTER}/scripts/build_v6_1_pool.py
echo "[done] distill_out/v6_1_pool.jsonl"

echo ""
echo "===== STEP 2b: build tool logs for v6.1 pool (uses new cache) ====="
python ${SPECMASTER}/scripts/build_tool_log.py \
    --in  ${SPECMASTER}/distill_out/v6_1_pool.jsonl \
    --out ${SPECMASTER}/distill_out/v6_1_tool_log.jsonl \
    --pred_cache /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json
echo "[done] distill_out/v6_1_tool_log.jsonl"

echo ""
echo "===== STEP 3: identify redistillation targets ====="
python ${SPECMASTER}/scripts/prepare_v6_1_redistill_indices.py
echo "[done] distill_out/v6_1_redistill_indices.json"
cat ${SPECMASTER}/distill_out/v6_1_redistill_summary.json

echo ""
echo "===== STEP 4: teacher distillation (GPT-5.5) for new/changed rows only ====="
mkdir -p ${SPECMASTER}/distill_out/v6_1
python ${SPECMASTER}/scripts/distill_teacher_cot.py \
    --model openai/specmaster/gpt-5.5 \
    --pool  ${SPECMASTER}/distill_out/v6_1_tool_log.jsonl \
    --only_indices ${SPECMASTER}/distill_out/v6_1_redistill_indices.json \
    --out_dir ${SPECMASTER}/distill_out/v6_1/ \
    --workers 12
# Output: distill_out/v6_1/distill_openai__specmaster__gpt-5.5.jsonl

echo ""
echo "===== STEP 5: build tool-grounded SFT trajectories ====="
python ${RECIPES}/teacher_cot_to_trajectory.py \
    --pool    ${SPECMASTER}/distill_out/v6_1_tool_log.jsonl \
    --distill ${SPECMASTER}/distill_out/v6_1/distill_openai__specmaster__gpt-5.5.jsonl \
              ${SPECMASTER}/distill_out/distill_openai__specmaster__gpt-5.5.jsonl \
    --out     ${SPECMASTER}/distill_out/trajectories_v6_1.jsonl \
    --rejects_out ${SPECMASTER}/distill_out/trajectories_v6_1_rejected.jsonl \
    --drop_provenance_violations
echo "[done] distill_out/trajectories_v6_1.jsonl"

echo ""
echo "===== STEP 5b: convert to LLaMA-Factory format ====="
python ${RECIPES}/trajectory_to_llamafactory.py \
    --in  ${SPECMASTER}/distill_out/trajectories_v6_1.jsonl \
    --out ${LLF}/data/nmr_matching_tool_sft_v6_1.json \
    --val_out ${LLF}/data/nmr_matching_tool_sft_v6_1_val.json \
    --val_frac 0.03
echo "[done] LLaMA-Factory/data/nmr_matching_tool_sft_v6_1.json"

echo ""
echo "===== STEP 5c: register dataset in dataset_info.json ====="
python - <<'PY'
import json
from pathlib import Path

info_path = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/dataset_info.json")
info = json.loads(info_path.read_text())

for name, fname in [
    ("nmr_matching_tool_sft_v6_1",     "nmr_matching_tool_sft_v6_1.json"),
    ("nmr_matching_tool_sft_v6_1_val", "nmr_matching_tool_sft_v6_1_val.json"),
]:
    if name not in info:
        info[name] = {
            "file_name": fname,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        }
        print(f"  registered: {name}")
    else:
        print(f"  already registered: {name}")

info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
print(f"[write] {info_path}")
PY

echo ""
echo "===== STEP 6: create training YAML ====="
cat > ${LLF}/examples/train_full/qwen3_32b_nmr_matching_tool_v6_1.yaml << 'YAML'
### Model
model_name_or_path: /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v5

### Method
stage: sft
do_train: true
finetuning_type: full

### Dataset
dataset: nmr_matching_tool_sft_v6_1
eval_dataset: nmr_matching_tool_sft_v6_1_val
template: qwen3
cutoff_len: 8192
max_samples: 999999
overwrite_cache: true
preprocessing_num_workers: 8

### Output
output_dir: saves/qwen3-32b-nmr-matching-tool/full-v6_1
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: true
save_total_limit: 3

### Train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000

### Eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100

### DeepSpeed
deepspeed: examples/deepspeed/ds_z3_config.json
YAML
echo "[done] examples/train_full/qwen3_32b_nmr_matching_tool_v6_1.yaml"

echo ""
echo "===== ALL PREP DONE — launch training with: ====="
echo "cd ${LLF} && FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 llamafactory-cli train examples/train_full/qwen3_32b_nmr_matching_tool_v6_1.yaml"

#!/bin/bash
# NMR Matching CoT SFT: 数据准备 + 训练 + 评估
# 在 8×A100 服务器上运行

set -e

LLAMA_FACTORY="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory"
SCRIPTS="/fs_mol/cxq/SpecMaster_jp/scripts"
EVAL_OUTPUT="/fs_mol/cxq/SpecMaster_jp/eval_results"

echo "============================================"
echo "Step 1: 数据准备"
echo "============================================"
python ${SCRIPTS}/prepare_cot_sft_data.py

echo ""
echo "============================================"
echo "Step 2: 训练 (LoRA, bf16, 8×A100)"
echo "============================================"
cd ${LLAMA_FACTORY}

# 8卡分布式训练
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
llamafactory-cli train examples/train_lora/qwen3_5_nmr_matching_lora.yaml

echo ""
echo "============================================"
echo "Step 3: 评估"
echo "============================================"
python ${SCRIPTS}/evaluate_nmr_matching.py \
    --model_path /fs_mol/cxq/SpecMaster_jp/models/Qwen3.5-35B-A3B \
    --adapter_path ${LLAMA_FACTORY}/saves/qwen3_5-nmr-matching/lora-exp1 \
    --test_file ${LLAMA_FACTORY}/data/nmr_matching_test.json \
    --output_dir ${EVAL_OUTPUT}

echo ""
echo "============================================"
echo "Done! Results at: ${EVAL_OUTPUT}"
echo "============================================"

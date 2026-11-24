#!/bin/bash
# v6.3 完整 test_2000 评测（新 prompt）

cd /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL

/fs_mol/cxq/conda_env/JPsft/bin/python3 recipe/nmr_matching_agent/standalone_infer_matching.py \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_3 \
  --from_test /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --n 2000 \
  --output /fs_mol/cxq/SpecMaster_jp/eval_results/v6_3_fixed_prompt_test2000_result.json \
  --max_turns 6 \
  --batch_size 8

echo "评测完成，结果保存至 eval_results/v6_3_fixed_prompt_test2000_result.json"

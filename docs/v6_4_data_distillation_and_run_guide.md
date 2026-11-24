# 谱图匹配：数据蒸馏、代码与运行指南

> 主仓库为`/fs_mol/cxq/SpecMaster_jp`，
> 另有两个同机依赖仓库：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL` 和 `/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory`。

## 任务说明

任务输入是候选分子的 SMILES、实验 `13C NMR` 峰和 `1H NMR` 峰；输出是候选结构是否匹配谱图的 `Yes/No`，并附带工具调用轨迹和证据。

## 仓库和核心代码

### SpecMaster_jp：数据处理、评测和运行包装

| 文件 | 用途 |
|---|---|
| `scripts/rewrite_final_evidence_from_tools.py` | 读取工具 observation，重写最终 `<think>` 和 `key_evidence` |
| `scripts/prepare_v6_4_sft_data.py` | 分组分层切分，并转成 LLaMA-Factory ShareGPT 工具格式 |
| `scripts/run_infer_parallel.sh` | 把评测集切成 GPU shards，调用单卡 Agent，再合并结果 |
| `scripts/_split_and_merge_infer.py` | 并行评测的切分与合并辅助脚本 |
| `scripts/summarize_matching_eval.py` | 汇总 accuracy、confusion matrix 和各负样本类型指标 |
| `scripts/compare_matching_evals.py` | 逐样本比较两个评测结果 |
| `scripts/run_nmr_matching_api.sh` | 启动当前 v6.4 HTTP API |
| `scripts/run_v6_1_pipeline.sh` | 上游 v6.1 数据、教师蒸馏和训练配置生成流程 |
| `scripts/run_v6_2_pipeline.sh` | 上游 v6.2 scorer 修复和选择性重蒸馏流程 |

### Open-AgentRL：轨迹、工具和推理实现

目录：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/`

| 文件 | 用途 |
|---|---|
| `teacher_cot_to_trajectory.py` | 将教师 CoT 和预计算工具日志合成为规范工具轨迹 |
| `cot_to_trajectory_v5.py` | 工具 observation 摘要、首轮数字清洗及数字 provenance 校验 |
| `trajectory_to_llamafactory.py` | Hermes 风格 `messages` 转 LLaMA-Factory `conversations` |
| `nmr_matching_runtime.py` | RDKit 计数、谱图匹配、官能团和逐峰匹配工具 Runtime |
| `standalone_infer_matching.py` | 单 GPU、多轮工具 Agent 推理入口 |
| `rule_based_router.py` | 高置信正样本规则路由，未命中时回退 32B 模型 |
| `evidence_reconstruction.py` | 根据真实工具轨迹重建模型最终证据，避免输出陈旧数字 |
| `prompt.py` | 系统提示词和输出约束 |
| `prediction_cache.json` | 候选结构预测谱缓存；`spectral_match` 的关键依赖 |
| `router_config.json` | 路由阈值、H 容差、模板语言和 assignment 配置 |

### LLaMA-Factory：SFT 数据、配置和权重

目录：`/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/`

| 文件 | 用途 |
|---|---|
| `data/nmr_matching_tool_sft_v6_4.json` | 实际训练数据，4009 条 |
| `data/nmr_matching_tool_sft_v6_4_val.json` | validation 数据，125 条 |
| `data/nmr_matching_tool_sft_v6_4_split_report.json` | 切分参数、SHA-256 和数量报告 |
| `data/dataset_info.json` | v6.4 train/val 的 LLaMA-Factory 注册项 |
| `examples/train_full/qwen3_32b_nmr_matching_tool_v6_4.yaml` | 实际训练配置 |
| `saves/qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63/` | 可用的最终权重 |


### 数据格式

源轨迹是一行一个 JSON：

```text
messages:
  system
  user
  assistant(<think> + <tool_call>)
  tool(real observation)
  ...
  assistant(<think> + <answer>)
meta:
  candidate_smiles / gt_smiles / label / negative_type / teacher / ...
```

`trajectory_to_llamafactory.py` 转成：

```text
conversations:
  system
  human
  [function_call, observation] * 4
  gpt(final think + answer)
tools: JSON 字符串形式的工具 schema
```

工具轨迹使用 `count_atoms`、`spectral_match` 和 `functional_group_scan`；其中 `count_atoms` 分别检查 C 和 H，所以每条共有 4 次调用。实际训练文件内嵌的 schema 有 `count_atoms`、`spectral_match`、`check_substructure`、`functional_group_scan` 四项。

## 评测集

位置：`data/nmr_matching_test_2000.json`。

总体组成：

- positive：800；
- negative：1200；

## 运行指令

```bash
# 使用已安装环境中的解释器和命令
export PATH=/fs_mol/cxq/conda_env/JPsft/bin:$PATH
cd /fs_mol/cxq/SpecMaster_jp
```

```bash
cd /fs_mol/cxq/SpecMaster_jp

bash scripts/run_infer_parallel.sh \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63 \
  --test_file data/nmr_matching_test_2000.json \
  --output eval_results/v6_4_router_recheck.json \
  --n_gpus 8 \
  --max_turns 6 \
  --use_router \
  --router_sim_threshold_both 0.85 \
  --router_sim_threshold_strict 0.92 \
  --router_template_variant_seed 0 \
  --router_skip_model_evidence
```


```bash
cd /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL

CUDA_VISIBLE_DEVICES=0 python recipe/nmr_matching_agent/standalone_infer_matching.py \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63 \
  --from_test /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --n 2 \
  --output /tmp/v6_4_two_cases.json \
  --max_turns 6 \
  --use_router \
  --router_sim_threshold_both 0.85 \
  --router_sim_threshold_strict 0.92 \
  --router_template_variant_seed 0 \
  --router_skip_model_evidence
```

### 7.6 启动 API

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash /fs_mol/cxq/SpecMaster_jp/scripts/run_nmr_matching_api.sh
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

请求示例见 `docs/nmr_matching_api.md`。默认 API 使用 `checkpoint-63`、单 worker，并在进程内串行执行模型推理。

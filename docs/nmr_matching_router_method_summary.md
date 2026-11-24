# NMR Matching：v6.3 工具 Agent + 双阈值规则路由方法总结

## 1. 当前方法概览

当前推理系统是一个混合架构：

1. **规则路由器先执行确定性工具**，识别证据充分的高置信正样本。
2. 命中规则的样本直接走 **template path**，输出 `Yes`，不加载或调用大模型。
3. 未命中规则的模糊或冲突样本走 **model path**，由 v6.3 Qwen3-32B 工具 Agent 最多进行 6 轮工具调用后判断 `Yes/No`。

核心目标是用确定性工具修复模型对高相似度正样本的误拒，同时让困难负样本继续交给模型判断。

主要文件：

- 并行推理：`/fs_mol/cxq/SpecMaster_jp/scripts/run_infer_parallel.sh`
- 单卡推理入口：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/standalone_infer_matching.py`
- 规则路由器：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/rule_based_router.py`
- 路由配置：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/router_config.json`
- 工具 Runtime：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/nmr_matching_runtime.py`
- 系统提示词：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prompt.py`
- 默认预测缓存：`/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json`
- 指标汇总：`/fs_mol/cxq/SpecMaster_jp/scripts/summarize_matching_eval.py`
- 两次实验比较：`/fs_mol/cxq/SpecMaster_jp/scripts/compare_matching_evals.py`

## 2. 输入与工具证据

测试集是 Alpaca 格式。`standalone_infer_matching.py` 从 `input` 文本中解析：

- candidate SMILES
- 实验 13C NMR 峰
- 实验 1H NMR 峰及积分
- `_label`
- `_negative_type`

Runtime 提供以下工具：

### 2.1 `count_atoms(C)`

RDKit 精确计算候选结构的碳原子数，并与实验 13C 独立信号数比较。

当前一致性定义：

```text
candidate_C_count >= experimental_13C_signal_count
```

候选碳数多于实验信号数可以由分子对称性或峰重叠解释；候选碳数少于实验信号数是硬冲突。

### 2.2 `count_atoms(H)`

RDKit 计算候选结构总氢数，与实验 1H 积分总和比较。template 路径允许绝对差不超过 `h_tolerance=2`；差值更大时进入 model 路径。

### 2.3 `spectral_match()`

从 prediction cache 获取候选结构的预测谱，再与实验谱计算：

- `similarity`：综合相似度
- `c_sim`：13C 子分数
- `h_sim`：1H 子分数

缓存缺失或预测失败时 `ok=false`，此时不会进入模板路径。

### 2.4 `functional_group_scan()`

使用 RDKit SMARTS 扫描候选结构中的已知官能团。它用于模板中的可解释证据，但当前不直接决定路由阈值。

### 2.5 模型路径的附加工具

模型工具 schema 还注册了 `check_substructure()` 和 `peak_assignment()`。其中 `peak_assignment()` 可在中等相似度样本中输出逐峰残差和未匹配实验峰；规则路由器本身不使用该工具。

## 3. 双阈值路由策略（Method A）

当前配置：

```json
{
  "sim_threshold_both": 0.85,
  "sim_threshold_strict": 0.92,
  "h_tolerance": 2,
  "template_variant_seed": 0,
  "template_languages": ["en", "zh", "ja"],
  "confidence_weights": {
    "sim": 0.40,
    "c_sim": 0.30,
    "h_sim": 0.30
  }
}
```

C/H 分别使用 `available` 和 `pass` 两个状态，缺失不等于不匹配。决策规则：

| C 状态 | H 状态 | 谱图条件 | 路径 |
|---|---|---|---|
| available + pass | available + pass | `spectral.ok && similarity >= 0.85` | template |
| missing | available + pass | `spectral.ok && similarity >= 0.92` | template |
| available + pass | missing | `spectral.ok && similarity >= 0.92` | template |
| missing | missing | 任意 | model |
| available + mismatch | 任意 | 任意 | model |
| 任意 | available + mismatch | 任意 | model |
| 任意 | 任意 | `spectral.ok=false` 或未达到对应阈值 | model |

模板路径只输出高置信 `Yes`，不存在模板 `No`。T1–T5 分别对应双通道精确、H 容差、C 信号减少、C 缺失和 H 缺失。H 积分存在时允许绝对差不超过 `h_tolerance=2`。每个场景有 `en/zh/ja` 三种措辞；使用包含完整样本身份、场景 ID 和 seed 的 SHA-256 稳定选择，因此跨进程可复现而不同样本具有语言多样性。完整 `<think>`、`<answer>` 和 transcript 都由实际工具结果确定性构造，不调用大模型。

模板路径的 confidence 为：

```text
0.40 * similarity + 0.30 * c_sim + 0.30 * h_sim
```

## 4. 历史阈值校准结果

以下数字来自修复 C/H missing/mismatch 对称性之前的旧门控，仅保留为阈值选择背景。当前规则将 available mismatch 强制送入 model，并让 C/H 单通道缺失对称使用 strict 阈值；修改后必须重新做独立校准，不能直接沿用下表作为当前覆盖率。

在 `nmr_matching_test_2000.json`（800 positive / 1200 negative）上的历史离线路由结果：

| 路由规则 | 正样本模板覆盖率 | 负样本误路由率 | similar_structure 误路由 |
|---|---:|---:|---:|
| 旧 flat `sim>=0.92` | 8.1% | 0.33%（4） | 1 |
| 历史 `both=0.85 / strict=0.92` | 25.2% | 0.83%（10） | 3 |
| 更激进 `both=0.80 / strict=0.92` | 27.1% | 1.33%（16） | 3 |

选择 0.85/0.92 的原因是：正样本覆盖率由 8.1% 提升到 25.2%，同时负样本误路由维持在 1% 以下。

注意：这里的“正样本模板覆盖率”不是端到端 positive recall，“负样本误路由率”也不是最终模型 FP。端到端收益取决于这些样本原本会被 v6.3 判对还是判错，必须通过相同模型、相同 prompt 的有/无路由对照实验测量。此外，阈值是在 test_2000 上校准的，因此这些数字属于开发集校准结果，不应当作完全独立的泛化评估。

## 5. 正确的 8 GPU 路由评估命令

应从 `SpecMaster_jp` 运行脚本，或者直接使用脚本绝对路径。不要在 `nmr_matching_agent` 目录下执行相对路径 `scripts/run_infer_parallel.sh`，因为该脚本位于 `SpecMaster_jp/scripts/`。

```bash
cd /fs_mol/cxq/SpecMaster_jp

bash scripts/run_infer_parallel.sh \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_3 \
  --test_file /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --output /fs_mol/cxq/SpecMaster_jp/eval_results/v6_3_fixed_prompt_router_085_092_test2000.json \
  --n_gpus 8 \
  --max_turns 6 \
  --use_router \
  --router_sim_threshold_both 0.85 \
  --router_sim_threshold_strict 0.92 \
  --router_template_variant_seed 0 \
  --router_skip_model_evidence
```

说明：

- 显式传入两个阈值，避免结果依赖运行时读取到的配置文件版本。
- `--router_skip_model_evidence` 直接使用路由器已经生成的结构化答案，明确保证 template path 零模型调用。
- 8 个进程分别绑定 GPU 0–7；测试集切成 8 个 shard，完成后自动合并。
- 运行前应激活含 RDKit、PyTorch、Transformers 和项目依赖的 `JPsft` 环境。
- 输出文件名使用新名字，避免覆盖已有的 `v6_3_fixed_prompt_test2000_result_router.json`。

如果当前目录必须保持在 `nmr_matching_agent`，可直接执行绝对路径：

```bash
bash /fs_mol/cxq/SpecMaster_jp/scripts/run_infer_parallel.sh \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_3 \
  --test_file /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --output /fs_mol/cxq/SpecMaster_jp/eval_results/v6_3_fixed_prompt_router_085_092_test2000.json \
  --n_gpus 8 \
  --max_turns 6 \
  --use_router \
  --router_sim_threshold_both 0.85 \
  --router_sim_threshold_strict 0.92 \
  --router_template_variant_seed 0 \
  --router_skip_model_evidence
```

## 6. 汇总端到端指标

```bash
cd /fs_mol/cxq/SpecMaster_jp

python scripts/summarize_matching_eval.py \
  --in eval_results/v6_3_fixed_prompt_router_085_092_test2000.json \
  --errors-out eval_results/v6_3_fixed_prompt_router_085_092_test2000_errors.csv \
  --show-errors 30
```

该命令输出：

- overall accuracy
- confusion matrix
- positive recall（`positive` 类型准确率，以及 confusion 中的 Yes→Yes）
- 各 `_negative_type` 准确率，包括 `similar_structure`
- 无法解析的输出数量
- 错误样本 CSV

## 7. 严格的有/无路由对照实验

为了只测路由器收益，baseline 必须使用同一 v6.3 权重、同一份当前 prompt、同一推理脚本和同一 `max_turns`；唯一差异是是否加 `--use_router`。

### 7.1 无路由 baseline

```bash
cd /fs_mol/cxq/SpecMaster_jp

bash scripts/run_infer_parallel.sh \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_3 \
  --test_file /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --output /fs_mol/cxq/SpecMaster_jp/eval_results/v6_3_fixed_prompt_no_router_test2000.json \
  --n_gpus 8 \
  --max_turns 6
```

### 7.2 与路由版本逐样本比较

```bash
python scripts/compare_matching_evals.py \
  --base eval_results/v6_3_fixed_prompt_no_router_test2000.json \
  --new eval_results/v6_3_fixed_prompt_router_085_092_test2000.json \
  --base-name v6.3-fixed-prompt \
  --new-name v6.3-router-0.85-0.92 \
  --show-flips 50
```

重点看：

- `new_only`：路由修复的样本数
- `base_only`：路由引入的新错误数
- `net_delta_new_minus_base`
- `pairwise_by_type` 中 positive 和 similar_structure 的净变化

不要直接把旧 `v6_3_test2000_result.json` 当作严格 baseline，除非确认它与当前路由实验使用的是同一版 prompt 和推理代码；否则比较会同时混入 prompt 变化。

## 8. 结果文件中的路由字段

每条结果包含：

- `_route`: `template` 或 `model`
- template 路径额外包含 `_template_id`、`_template_variant_id`、`_template_language`、`_template_variant_seed`、`_template_variant_digest`、`_c_available`、`_h_available`、`_c_match`、`_h_match`、`_confidence`、`_skip_model`
- `_label`
- `_negative_type`
- `answer`
- `tool_sequence`
- `transcript`
- `fail_reason`

因此可以进一步统计：

- template/model 各自的样本量和准确率
- template precision 与正样本覆盖率
- template 误路由的负样本类型
- 模型路径剩余错误的类别分布
- 模型调用减少比例

## 9. 当前边界与注意事项

1. **路由器只能修复一部分高置信正样本。** 大量正样本错误位于中低相似度区间，或者模型未充分调用计数工具，不能通过继续降低模板阈值安全解决。
2. **C 一致是单向约束。** `候选 C 数 >= 13C 信号数` 并不证明结构正确，只排除了“候选碳数少于实验峰数”的硬冲突。
3. **H 一致允许 ±2。** 这是对未报告 OH/NH 等可交换氢和实验积分误差的容忍；差值超过 2 才视为模板硬冲突。
4. **模板路径只预测 Yes。** 被错误送入模板的负样本必然成为 FP，因此阈值应优先控制负样本误路由。
5. **prediction cache 是关键依赖。** cache miss 会使 `spectral_match.ok=false`，路由器自动回退模型路径；模型路径若调用相同工具，可能触发远程预测。
6. **官能团扫描不参与当前门控。** 它仅增强解释文本，不能抵消低谱图相似度或原子数冲突。
7. **完整评估必须同时报告准确率与成本。** 除 overall、positive recall、similar_structure 外，还应报告 template 比例，即减少了多少次 32B 模型推理。
8. **当前工具清单存在文档一致性检查点。** 推理 schema 注册了 `peak_assignment`，但 `prompt.py` 的文字仍将 `TOOL_DOC_FULL` 描述为 4-tool 文档；后续修改 prompt 或重训时应统一核对。

## 10. 新 session 建议起点

新 session 可直接引用本文件，并从以下问题继续：

1. 先完成同一 v6.3 fixed prompt 的 no-router/router 对照，得到实际净增益。
2. 统计 template 路径的数量、precision、positive coverage 以及按负样本类型的 FP。
3. 分析 model path 中剩余的正样本 FN，按 `similarity`、`c_sim`、`h_sim`、工具调用序列和 peak assignment 覆盖率分桶。
4. 在独立验证集上复核 0.85/0.92，避免 test_2000 阈值校准偏差。
5. 再决定是否为中相似度区间增加负向规则、逐峰 assignment 规则或学习型路由器。

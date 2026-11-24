# RSI 第 00 步：本地版本基线

本目录记录谱图匹配任务在 2026-11-24 的可追溯基线。这一步只做版本管理、上传范围和凭证读取方式，不改匹配 prompt、Router 阈值、工具评分、数据标签或模型路径，也不启动训练或推理。

## 组件

| 组件 | 路径 | 角色 |
|---|---|---|
| SpecMaster | `/fs_mol/cxq/SpecMaster_jp` | 数据处理、评测包装、API 启动脚本、本基线 |
| Open-AgentRL | `/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL` | 匹配 Agent、工具 Runtime、Router |
| LLaMA-Factory | `/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory` | SFT 配置与 checkpoint-63。本步不改其工作区 |

对应关系见 `baseline_manifest.json`。该文件不记录“包含它自己的那次提交”。SpecMaster 与 Open-AgentRL 的最终 commit 写在未跟踪的 `rsi_runs/step_00_*/handoff.json`。

## 跟踪与排除

SpecMaster 跟踪 `scripts/`、`perturbation_generator/` 源码、当前部署说明、`configs/llamafactory/`、`docs/rsi/`、`.env.example` 和 `tests/fixtures/` 中的小型合成样例。

默认不跟踪模型、`data/`、`cot_output*`、`distill_out*`、评测输出、负样本与扰动产物、预测缓存、`smoke_test/`、`rsi_runs/`、日志、`node_modules`、Python 缓存和 IDE 配置。不要使用全局 `*.json` 忽略规则。数据和模型只在 manifest 里登记路径、大小和哈希。

Open-AgentRL 只追加 `recipe/nmr_matching_agent/` 的源码、`router_config.json`、`tool_config_matching.yaml`、测试和说明，以及 `verl/tools/nmr_matching_tools.py`。该目录的 `.gitignore` 只排除 `data*`、`prediction_cache*`、日志和结果文件。原有无关工作区改动保持未暂存。

## 凭证

教师蒸馏和闭源评测脚本从环境变量 `NMR_GATEWAY_API_KEY` 读取网关密钥。变量缺失或只有空白时抛出 `RuntimeError`，不再回退到旧的字面量。`.env.example` 只有变量名和空值。真实值留在未跟踪的本地环境或 `rsi_runs` 原始备份中。

## 私有远端

不要把本步提交推到原来的上游：

- Open-AgentRL 的 `origin` 仍是 `https://github.com/Gen-Verse/Open-AgentRL.git`
- LLaMA-Factory 的 `origin` 仍是 `https://github.com/hiyouga/LLaMA-Factory.git`

本步不创建远端、不推送。用户自行创建私有远端后，把下面的 `<PRIVATE_REMOTE_URL>` 换成实际地址。

若索引已暂存但尚未提交，先设置真实的 `user.name` 和 `user.email`，再只提交已暂存内容。不要使用 `git add -A` 或 `git commit -a`。

```bash
git -C /fs_mol/cxq/SpecMaster_jp commit -m "Establish a local baseline for the NMR matching workflow."
git -C /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL commit -m "Track the NMR matching agent on a local prep branch."

git -C /fs_mol/cxq/SpecMaster_jp remote add origin <PRIVATE_REMOTE_URL>
git -C /fs_mol/cxq/SpecMaster_jp push -u origin main

git -C /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL remote add private <PRIVATE_REMOTE_URL>
git -C /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL push -u private rsi/prep-20261124
```

Open-AgentRL 推送的是本地分支 `rsi/prep-20261124`，父提交为当前本地 `HEAD`，不包含尚未合并的 `origin/main`。模型分片、训练 JSON 和预测缓存仍留在服务器上。

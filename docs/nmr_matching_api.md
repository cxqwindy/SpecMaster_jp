# NMR Matching API（单张 80G A100）

## 启动

```bash
CUDA_VISIBLE_DEVICES=0 bash /fs_mol/cxq/SpecMaster_jp/scripts/run_nmr_matching_api.sh
```

服务只允许一个 worker，并使用进程内锁串行执行 32B 推理。模型、prediction cache 和工具 manager 均只在启动时加载一次。

## 接口

- `GET /health` → `{"ready": true}`
- `POST /match`

请求字段 `include_model_reasoning_in_key_evidence`：

- `false`（默认）：key_evidence 只包含确定性工具证据；template 请求保持零模型调用。
- H 积分比较默认使用 `NMR_ROUTER_H_TOLERANCE=2`，该容差同时传入 Router 和模型工具 Runtime。
- `true`：把模型最终 `<think>` 放在确定性工具 key_evidence 前面。template 请求复用 Router 已有的四个 observation，执行一次不开放工具的模型生成；最终 Yes/No 仍由 template 决定。model 请求保持原多轮 Agent 并复用自身轨迹。

公开响应仅包含 `final_answer`、grounded `final_think`、`key_evidence` 和逐轮 `trajectory`。不返回 confidence、route、template ID、内部 metadata 或原始模型 `<answer>`。

```bash
curl -s http://127.0.0.1:8000/match \
  -H 'Content-Type: application/json' \
  -d '{
    "candidate_smiles": "CCO",
    "c_nmr_peaks": [{"delta (ppm)": 58.0}, {"delta (ppm)": 18.0}],
    "h_nmr_peaks": [{"delta": 3.6, "nH": 2}, {"delta": 1.2, "nH": 3}],
    "nmr_solvent": "CDCl3",
    "nmr_frequency": "400 MHz",
    "include_model_reasoning_in_key_evidence": false
  }' | python -m json.tool
```

如部署到公网，应在外层反向代理增加 TLS、鉴权、请求体大小限制和超时。

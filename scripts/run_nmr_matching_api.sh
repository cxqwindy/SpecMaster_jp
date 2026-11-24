#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL${PYTHONPATH:+:$PYTHONPATH}"
export NMR_MATCHING_MODEL_PATH="${NMR_MATCHING_MODEL_PATH:-/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63}"
export NMR_MATCHING_MAX_TURNS="${NMR_MATCHING_MAX_TURNS:-6}"
export NMR_ROUTER_SIM_BOTH="${NMR_ROUTER_SIM_BOTH:-0.85}"
export NMR_ROUTER_SIM_STRICT="${NMR_ROUTER_SIM_STRICT:-0.92}"
export NMR_ROUTER_H_TOLERANCE="${NMR_ROUTER_H_TOLERANCE:-2}"
export NMR_ROUTER_TEMPLATE_SEED="${NMR_ROUTER_TEMPLATE_SEED:-0}"
export NMR_MATCHING_ALLOW_REMOTE_PREDICT="${NMR_MATCHING_ALLOW_REMOTE_PREDICT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

HOST="${NMR_API_HOST:-0.0.0.0}"
PORT="${NMR_API_PORT:-8000}"

cd /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL
exec /fs_mol/cxq/conda_env/JPsft/bin/uvicorn \
  recipe.nmr_matching_agent.api_service:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 120

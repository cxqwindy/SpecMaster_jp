#!/usr/bin/env python3
"""Evaluate NMR match judgment on the real-conversation dataset (real_task_clean.jsonl).

No ground-truth labels — this compares each model's Yes/No verdict + reasoning.
Runs one model per invocation (SFT via vLLM, or an API model via LiteLLM) and
writes per-record verdicts to a per-model JSONL. A separate --compare step
merges all model outputs into a side-by-side table + agreement stats.

Usage:
  # SFT model (vLLM, OpenAgentRL env):
  python evaluate_real_task.py --backend vllm \
      --model_path saves/qwen3-32b-nmr-matching/full-exp3/checkpoint-200 \
      --tag exp3 --tokenizer_path /fs_mol/cxq/SpecMaster_jp/models/Qwen3-32B

  # Gemini (API):
  python evaluate_real_task.py --backend api \
      --model openai/specxmaster/gemini-3.1-pro-preview --tag gemini

  # Merge & compare:
  python evaluate_real_task.py --compare
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/fs_mol/cxq/SpecMaster_jp/scripts")
from evaluate_rerank_fairnmr500 import INSTRUCTION, format_input, parse_yes_conf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key


DATA = "/fs_mol/cxq/SpecMaster_jp/data/real_task_clean.jsonl"
OUT_DIR = Path("/fs_mol/cxq/SpecMaster_jp/eval_results_real_task")

SYSTEM = (
    "You are an NMR spectrum-structure matching expert. Given NMR data and a "
    "candidate SMILES, determine whether they match. You do NOT have the target "
    "molecular formula. Output ONLY a JSON object: "
    '{"final_answer": "Yes" or "No", "confidence": <float 0.0-1.0>, '
    '"key_evidence": "<one sentence>"}'
)
API_BASE = "https://ai-gateway-global.dp.tech"


def load_records():
    return [json.loads(l) for l in open(DATA)]


def build_prompt_text(rec):
    return INSTRUCTION + "\n\n" + format_input(
        rec["candidate_smiles"], rec["c_nmr_peaks"], rec["h_nmr_peaks"],
        rec.get("nmr_solvent", "") or "")


def run_vllm(args, records):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prompts = []
    for rec in records:
        msg = [{"role": "user", "content": build_prompt_text(rec)}]
        prompts.append(tok.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True,
              dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    results = []
    for rec, o in zip(records, outs):
        text = o.outputs[0].text
        ans, conf = parse_yes_conf(text)
        results.append({"id": rec["id"], "ans": ans, "conf": conf,
                        "raw": text[:600]})
    return results


def run_api(args, records):
    api_key = require_gateway_api_key()
    import litellm
    from litellm import completion
    from concurrent.futures import ThreadPoolExecutor, as_completed
    litellm.drop_params = True

    def one(rec):
        for attempt in range(6):
            try:
                resp = completion(
                    model=args.model, api_key=api_key, api_base=API_BASE,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": build_prompt_text(rec)}],
                    max_tokens=16384, temperature=0.0, timeout=180)
                text = resp.choices[0].message.content or ""
                if not text and hasattr(resp.choices[0].message, "reasoning_content"):
                    text = resp.choices[0].message.reasoning_content or ""
                ans, conf = parse_yes_conf(text)
                if ans == "Unknown" and attempt < 5:
                    import time; time.sleep(2 * (attempt + 1)); continue
                return {"id": rec["id"], "ans": ans, "conf": conf, "raw": text[:600]}
            except Exception:
                import time; time.sleep(3 * (attempt + 1))
        return {"id": rec["id"], "ans": "Unknown", "conf": 0.5, "raw": ""}

    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, r): i for i, r in enumerate(records)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


def compare():
    records = {r["id"]: r for r in load_records()}
    # load all per-model verdict files
    models = {}
    for f in sorted(OUT_DIR.glob("verdicts_*.jsonl")):
        tag = f.stem.replace("verdicts_", "")
        models[tag] = {v["id"]: v for v in (json.loads(l) for l in open(f))}
    if not models:
        print("No verdict files found in", OUT_DIR); return
    tags = list(models.keys())
    print(f"Models: {tags}\n")

    # side-by-side
    from collections import Counter
    dist = {t: Counter() for t in tags}
    agree = 0
    rows = []
    for rid in sorted(records):
        verdicts = {t: models[t].get(rid, {}).get("ans", "-") for t in tags}
        for t in tags:
            dist[t][verdicts[t]] += 1
        vals = [verdicts[t] for t in tags if verdicts[t] in ("Yes", "No")]
        all_agree = len(set(vals)) <= 1 and len(vals) == len(tags)
        if all_agree:
            agree += 1
        rows.append((rid, verdicts, records[rid]["candidate_smiles"][:40]))

    print("=== per-model answer distribution ===")
    for t in tags:
        print(f"  {t}: {dict(dist[t])}")
    print(f"\n=== agreement (all models same Yes/No): {agree}/{len(records)} ===\n")

    print("=== per-record ===")
    hdr = "id  " + "  ".join(f"{t:>10}" for t in tags) + "   candidate"
    print(hdr)
    for rid, v, smi in rows:
        line = f"{rid:<3} " + "  ".join(f"{v[t]:>10}" for t in tags) + f"   {smi}"
        # mark disagreements
        vals = [v[t] for t in tags if v[t] in ("Yes", "No")]
        mark = " " if (len(set(vals)) <= 1 and len(vals) == len(tags)) else " *"
        print(line + mark)

    # save merged
    merged = []
    for rid, v, smi in rows:
        merged.append({"id": rid, "candidate_smiles": records[rid]["candidate_smiles"],
                       "verdicts": v})
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_DIR}/comparison.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "api"])
    ap.add_argument("--model_path")            # vllm
    ap.add_argument("--tokenizer_path", default="/fs_mol/cxq/SpecMaster_jp/models/Qwen3-32B")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--model")                 # api model id
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", help="short name for output file")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare(); return

    records = load_records()
    print(f"Loaded {len(records)} real-task records")
    if args.backend == "vllm":
        results = run_vllm(args, records)
    else:
        results = run_api(args, records)

    outp = OUT_DIR / f"verdicts_{args.tag}.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = Counter(r["ans"] for r in results)
    print(f"[{args.tag}] verdict dist: {dict(dist)}")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()

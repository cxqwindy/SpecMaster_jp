"""从 v5_eval_test2000.json 里挑出 v5 答对的 case,拉出 peak 数据,
包装成 call_v5_vllm.py 可用的测试用例。

- eval 输出: /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json
- 源 peaks:  /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json
- 输出:      /fs_mol/cxq/SpecMaster_jp/data/verify_cases_v5.json
"""
import json
import re
from pathlib import Path
from collections import Counter

EVAL = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json"
TEST = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
OUT = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_v5.json"

_C_NMR_RE = re.compile(r"13C NMR[^:]*:\s*([^\n]+)", re.IGNORECASE)
_H_NMR_RE = re.compile(r"1H NMR[^:]*:\s*([^\n]+)", re.IGNORECASE)
_H_PEAK_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*\(([^)]*)\)")
_FREQ_SOLV_RE = re.compile(r"\(([\d.]+)\s*MHz,\s*([A-Za-z0-9\-]+)\)")


def parse_input(input_text: str):
    """从 alpaca 的 input 字符串里解析出 c_peaks / h_peaks / freq / solvent。"""
    c_peaks, h_peaks = [], []
    freq, solvent = 400.0, "CDCl3"
    cm = _C_NMR_RE.search(input_text)
    if cm:
        m = _FREQ_SOLV_RE.search(cm.group(0))
        if m:
            try:
                freq = float(m.group(1))
                solvent = m.group(2)
            except Exception:
                pass
        if "N/A" not in cm.group(1):
            for tok in cm.group(1).split(","):
                tok = tok.strip()
                try:
                    c_peaks.append({"delta (ppm)": float(tok)})
                except ValueError:
                    continue
    hm = _H_NMR_RE.search(input_text)
    if hm and "N/A" not in hm.group(1):
        for delta_s, inner in _H_PEAK_RE.findall(hm.group(1)):
            try:
                delta = float(delta_s)
            except ValueError:
                continue
            nH = None
            m_nh = re.search(r"(\d+)\s*H", inner)
            if m_nh:
                nH = int(m_nh.group(1))
            h_peaks.append({"delta": delta, "centroid": delta, "nH": nH})
    return c_peaks, h_peaks, freq, solvent


def parse_final(answer_str: str):
    try:
        obj = json.loads(answer_str)
        return obj.get("final_answer")
    except Exception:
        m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', answer_str or "")
        return m.group(1) if m else None


def main():
    ev = json.load(open(EVAL))
    tt = json.load(open(TEST))
    assert len(ev) == len(tt), f"len mismatch {len(ev)} vs {len(tt)}"

    # 按 negative_type 分桶, 记录 v5 答对的 index
    correct_by_type = {}
    for idx, (e, t) in enumerate(zip(ev, tt)):
        gt = t["output"]
        pred = parse_final(e["answer"])
        if pred is None or pred != gt:
            continue
        if e.get("fail_reason"):
            continue
        # 严格:要求有实际工具调用(过滤掉 answer-first 的偷懒 case)
        tool_seq = [x for x in (e.get("tool_sequence") or []) if x and x != "none"]
        if not tool_seq:
            continue
        nt = t["_negative_type"]
        correct_by_type.setdefault(nt, []).append((idx, e, t, tool_seq))

    print("v5 答对且有真实工具调用的池:")
    for nt in sorted(correct_by_type):
        print(f"  {nt:<60} {len(correct_by_type[nt])}")

    # 每类取 1-2 条,构造输出
    TARGETS = [
        ("positive", 3),
        ("neg/same_formula", 2),
        ("neg/similar_structure", 2),
        ("neg/random", 1),
        ("neg/hard_nmr/alkyl_chain_shift/insert_ch2", 1),
        ("neg/perturbation/alkyl_chain_shift/insert_ch2", 1),
        ("neg/perturbation/heteroatom_replacement/O_to_S", 1),
        ("neg/perturbation/functional_group_replacement/alcohol_to_thiol", 1),
    ]

    out = []
    for nt, want in TARGETS:
        pool = correct_by_type.get(nt, [])
        # 优先挑 turns 3-5 的(既真正用了工具又不过分冗长)
        pool.sort(key=lambda x: (len(x[3]) < 2, len(x[3]) > 6, len(x[3])))
        picks = pool[:want]
        for idx, e, t, tool_seq in picks:
            c_peaks, h_peaks, freq, solvent = parse_input(t["input"])
            out.append({
                "index": idx,
                "negative_type": nt,
                "gt_answer": t["output"],
                "candidate_smiles": t["_candidate_smiles"],
                "c_nmr_peaks": c_peaks,
                "h_nmr_peaks": h_peaks,
                "nmr_solvent": solvent,
                "nmr_frequency": freq,
                "_v5_tool_sequence": tool_seq,
                "_v5_answer_preview": (e["answer"] or "")[:200],
            })
        print(f"  [{nt}] want={want} got={len(picks)}")

    Path(OUT).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n写入 {OUT}, 共 {len(out)} 条")


if __name__ == "__main__":
    main()

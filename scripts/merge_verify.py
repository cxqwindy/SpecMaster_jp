"""合并 verify_cases.json (8 条,step180 历史输出对齐)
和 verify_cases_expanded.json (26 条,当前部署稳定基线) 为一个总集。

按 index 去重(理论上无重叠,构建 expanded 时已 exclude 了原 8 条 index)。
输出: /fs_mol/cxq/SpecMaster_jp/data/verify_cases_all.json
"""
import json

A = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases.json"
B = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_expanded.json"
OUT = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_all.json"

a = json.load(open(A))
b = json.load(open(B))
print(f"verify_cases.json:          {len(a)} 条")
print(f"verify_cases_expanded.json: {len(b)} 条")

seen = set()
merged = []
for src, name in [(a, "base"), (b, "expanded")]:
    for c in src:
        key = c["index"]
        if key in seen:
            print(f"  跳过重复 index={key} (来自 {name})")
            continue
        seen.add(key)
        merged.append(c)

# 排序:先 positive,再各 neg 类型,组内按 index
def sort_key(c):
    order = {
        "positive": 0,
        "neg/same_formula": 1,
        "neg/similar_structure": 2,
        "neg/random": 3,
        "neg/hard_nmr/alkyl_chain_shift/insert_ch2": 4,
        "neg/perturbation/alkyl_chain_shift/insert_ch2": 5,
        "neg/perturbation/heteroatom_replacement/O_to_S": 6,
        "neg/perturbation/functional_group_replacement/alcohol_to_thiol": 7,
    }
    return (order.get(c["negative_type"], 99), c["index"])

merged.sort(key=sort_key)
json.dump(merged, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\n合并完成: {len(merged)} 条 -> {OUT}")

# 按类型统计
from collections import Counter
cnt = Counter(c["negative_type"] for c in merged)
print("\n每类分布:")
for nt, n in sorted(cnt.items(), key=lambda x: sort_key({"negative_type": x[0], "index": 0})):
    print(f"  {nt:<60} {n}")

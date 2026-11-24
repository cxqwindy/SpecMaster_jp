import json

P = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_expanded.json"
cases = json.load(open(P))
before = len(cases)
cases = [c for c in cases if c["index"] != 963]
after = len(cases)
json.dump(cases, open(P, "w"), indent=1, ensure_ascii=False)
print(f"drop idx=963: {before} -> {after}")

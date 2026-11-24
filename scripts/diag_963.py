import json
import httpx

cases = json.load(open("/fs_mol/cxq/SpecMaster_jp/data/verify_cases_expanded.json"))
c = [x for x in cases if x["index"] == 963][0]

r = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "nmr-matching",
        "messages": [{"role": "user", "content": c["user_content"]}],
        "temperature": 0,
        "max_tokens": 2900,
    },
    timeout=300,
).json()

if "choices" not in r:
    print("BAD RESPONSE:", json.dumps(r, indent=2)[:500])
else:
    ch = r["choices"][0]
    print("finish_reason:", ch.get("finish_reason"))
    print("completion_tokens:", r["usage"]["completion_tokens"])
    content = ch["message"]["content"] or ""
    print(f"content length (chars): {len(content)}")
    print("--- content tail 400 ---")
    print(content[-400:])
    print("--- content head 200 ---")
    print(content[:200])

import json, random

SRC = "eval_public (1).jsonl"
N = 90
Q_KEYS = ["question", "query", "prompt"]
A_KEYS = ["answer", "gold_answer", "expected_answer", "ground_truth"]

def pick(row, keys):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
print(f"{len(rows)} rows | keys: {list(rows[0].keys())}")

random.seed(7)
sample = random.sample(rows, min(N, len(rows)))
out = []
for i, r in enumerate(sample, 1):
    q = pick(r, Q_KEYS)
    if q is None:
        raise SystemExit(f"No question field. Keys are {list(r.keys())} -> edit Q_KEYS.")
    out.append({"id": f"q{i}", "query": q, "gold": pick(r, A_KEYS), "raw": r})

json.dump(out, open("eval_set.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"wrote eval_set.json with {len(out)} queries")
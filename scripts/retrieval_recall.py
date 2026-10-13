"""Retrieval recall on the diag question set: officeqa (v1) vs officeqa_v2.
docHit@K = #retrieved chunks (top K) from a gold source file; docAny@K = any;
ansLiteral = gold answer string appears in the rendered context."""
import json
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OFFICEQA_LIMIT", "20")
from harness.benchmarks.officeqa import OfficeQAAdapter
from harness.benchmarks.officeqa_v2 import OfficeQAAdapterV2

diag = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/workspace/officeqa-diag/results.json"))
uids = [d["uid"] for d in diag]
v1, v2 = OfficeQAAdapter(source="hf"), OfficeQAAdapterV2(source="hf")
out = []
for name, b, ranker in (("v1", v1, lambda q: v1._bm25.top(q, len(v1._chunks))), ("v2", v2, v2.rank)):
    rows = {r["uid"]: r for r in b.rows}
    by_uid = {b.uid(t): t for t in b.train_tasks() + b.test_tasks()}
    for u in uids:
        r = rows[u]; gold = {n.strip() for n in re.split(r"[,\n;]", r["source_files"]) if n.strip()}
        rk = ranker(r["question"])
        ans = str(r["answer"]).replace("%", "").replace(",", "").strip()
        ctx = by_uid[u].replace(",", "")
        out.append({"adapter": name, "uid": u,
                    "docHit@20": sum(b._chunk_src[i] in gold for i in rk[:20]),
                    "docAny@20": any(b._chunk_src[i] in gold for i in rk[:20]),
                    "firstGoldRank": next((n + 1 for n, i in enumerate(rk) if b._chunk_src[i] in gold), None),
                    "ansLiteralInContext": ans in ctx, "ctx_chars": len(by_uid[u])})
for name in ("v1", "v2"):
    rs = [o for o in out if o["adapter"] == name]
    print(name, "docAny@20 %d/%d" % (sum(o["docAny@20"] for o in rs), len(rs)),
          "mean docHit@20 %.1f" % (sum(o["docHit@20"] for o in rs) / len(rs)),
          "ansLiteral %d/%d" % (sum(o["ansLiteralInContext"] for o in rs), len(rs)),
          "mean ctx chars %d" % (sum(o["ctx_chars"] for o in rs) / len(rs)))
Path("logs").mkdir(exist_ok=True)
json.dump(out, open("logs/retrieval_recall.json", "w"), indent=1)
for o in out:
    print(o)

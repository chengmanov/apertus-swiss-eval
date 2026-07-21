"""Retrieval-recall diagnostic: for each eval item, does BM25 put the gold unit
in the top-k? recall@k is the *citation ceiling* — arm D citation cannot exceed it.
Also reports recall@1/5/8/20 so we can see reranking headroom.

Usage: python recall_diag.py <repo_dir> <module_name>
(module_name e.g. dsg_eval, hmg_eval, vvg_eval, prsg_eval, legal_eval)
"""
import json, sys
from pathlib import Path

repo = Path(sys.argv[1]); mod = sys.argv[2]
sys.path.insert(0, str(repo / "src"))
from importlib import import_module
corpus = import_module(f"{mod}.corpus"); retr = import_module(f"{mod}.retrieval")

items = [json.loads(l) for l in (repo / "tasks" / "eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
langs = sorted({i["lang"] for i in items})
R = {lg: retr.Retriever(corpus.load_chunks(lg)) for lg in langs}
KS = [1, 5, 8, 20]
hits = {k: 0 for k in KS}
n = 0
for it in items:
    ranked = R[it["lang"]].top(it["question"], k=max(KS))
    got = [c.article.lower() for c in ranked]
    gold = {str(a).lower() for a in it["gold_article"]} | {str(a).lower() for a in it.get("also_acceptable_article", [])}
    n += 1
    for k in KS:
        if gold & set(got[:k]):
            hits[k] += 1
name = mod.replace("_eval", "").upper()
print(f"{name:6} n={n}  " + "  ".join(f"recall@{k}={100*hits[k]/n:5.1f}%" for k in KS))

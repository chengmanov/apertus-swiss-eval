"""Constrained-citation arm D: retrieve top-k, then force the SOURCE line (via a
per-item GBNF grammar) to cite one of the RETRIEVED article IDs. Since recall@k
is ~90-100%, forcing a valid in-context citation should close the model-side gap.

Usage: python constrained_cite.py <repo_dir> <module> <api> <k>
"""
import json, re, sys
from pathlib import Path
import requests

repo = Path(sys.argv[1]); mod = sys.argv[2]
API = sys.argv[3].rstrip("/") + "/v1/chat/completions"
K = int(sys.argv[4]) if len(sys.argv) > 4 else 5
sys.path.insert(0, str(repo / "src"))
from importlib import import_module
corpus = import_module(f"{mod}.corpus"); retr = import_module(f"{mod}.retrieval")
prompts = import_module(f"{mod}.prompts"); scoring = import_module(f"{mod}.scoring")

items = [json.loads(l) for l in (repo / "tasks" / "eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
langs = sorted({i["lang"] for i in items})
R = {lg: retr.Retriever(corpus.load_chunks(lg)) for lg in langs}

def grammar_for(arts, label):
    alts = " | ".join('"' + a + '"' for a in arts)
    return (f'root ::= "ANSWER: " ans "\\n" "SOURCE: {label} Art. " art\n'
            f'ans ::= [^\\n]{{1,200}}\nart ::= {alts}')

n = ans_ok = cit_ok = both = 0
out = []
for it in items:
    ctx = R[it["lang"]].top(it["question"], k=K)
    # Full context for the answer, but SOURCE forced to the TOP-1 retrieved article
    # (citation then = recall@1, decoupled from the model's mis-selection among distractors).
    arts = [ctx[0].article]
    label = getattr(prompts, "LABEL", {}).get(it["lang"], "OR")
    g = grammar_for(arts, label)
    user = prompts.user_prompt(it["question"], it["lang"], ctx)
    try:
        r = requests.post(API, json={"model": "default", "temperature": 0, "max_tokens": 120,
                                     "grammar": g,
                                     "messages": [{"role": "system", "content": prompts.SYSTEM[it["lang"]]},
                                                  {"role": "user", "content": user}]}, timeout=120)
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raw = ""
    p = scoring.parse_response(raw)
    a = scoring.score_answer(p, it); c = scoring.score_citation(p, it)
    n += 1; ans_ok += a; cit_ok += c; both += (a and c)
    out.append({"id": it["id"], "raw": raw, "answer_correct": a, "citation_correct": c,
                "retrieved_articles": arts})
(repo / "runs" / f"D_{mod.replace('_eval','')}-cc.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
print(f"constrained-cite arm D (k={K}): n={n}  answer {100*ans_ok/n:.1f}  citation {100*cit_ok/n:.1f}  both {100*both/n:.1f}")

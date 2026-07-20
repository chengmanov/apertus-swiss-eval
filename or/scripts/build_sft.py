"""Build the SFT dataset from tasks/sft_drafts/.

Contamination control is enforced HERE, independently of the authoring
instructions: any item whose gold_article/also_acceptable_article touches
tasks/sft_exclusions.json (the set of articles cited as gold in the eval set)
is rejected, and any item whose normalized question collides with an eval
question is rejected.

RETRIEVAL-AWARE: for each item we emit two training examples — one plain
(question only) and one with retrieved context (top-k passages, with the gold
article guaranteed present) — so the fine-tune keeps its ANSWER/SOURCE
discipline both without retrieval (arm C) and, crucially, *with* it (arm D).
Training question-only and serving with RAG is a distribution mismatch that
degrades citation under retrieval; this matches training to inference.

Output: tasks/sft_train.jsonl, tasks/sft_val.jsonl (95/5, deterministic).
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_eval.corpus import load_chunks
from legal_eval.prompts import SYSTEM, user_prompt
from legal_eval.retrieval import Retriever
from legal_eval.scoring import normalize

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "tasks" / "sft_drafts"
EXCL = json.loads((ROOT / "tasks" / "sft_exclusions.json").read_text(encoding="utf-8"))
EVAL = ROOT / "tasks" / "eval_v1.jsonl"
EXCLUDED_ARTICLES = {str(a).lower() for a in EXCL.get("articles", [])}
# Training context is deliberately shorter than eval retrieval (k=5): a compact
# gold+distractor context is enough to teach "keep citing under retrieval", and
# short sequences keep training tractable given the slow Python xIELU fallback.
RAG_K = 2


def assistant_target(item: dict) -> str:
    gold = item["gold"]
    answer = "; ".join(gold) if isinstance(gold, list) else str(gold)
    arts = ", ".join(f"Art. {a}" for a in item["gold_article"])
    return f"ANSWER: {answer}\nSOURCE: OR {arts}"


def main() -> int:
    eval_questions = {normalize(json.loads(l)["question"])
                      for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()}
    kept: list[dict] = []
    rejected = Counter()
    seen: set[str] = set()

    for path in sorted(DRAFTS.glob("*.json")):
        _, _, lang = path.stem.rpartition("_")
        for item in json.loads(path.read_text(encoding="utf-8")):
            item.setdefault("lang", lang)
            arts = {str(a).lower() for a in item.get("gold_article", [])} | \
                   {str(a).lower() for a in item.get("also_acceptable_article", [])}
            if not arts or not item.get("question") or item.get("gold") in (None, "", []):
                rejected["malformed"] += 1
                continue
            if arts & EXCLUDED_ARTICLES:
                rejected["contaminated"] += 1
                continue
            q = normalize(item["question"])
            if q in eval_questions:
                rejected["eval-collision"] += 1
                continue
            if q in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(q)
            item["gold_article"] = [str(a) for a in item["gold_article"]]
            kept.append(item)

    # Retrieval indexes + article lookup per language (for the RAG-aware variant).
    langs = sorted({i["lang"] for i in kept})
    chunks_by_lang = {lg: load_chunks(lg) for lg in langs}
    retrievers = {lg: Retriever(chunks_by_lang[lg]) for lg in langs}
    art_lookup = {lg: {c.article.lower(): c for c in chunks_by_lang[lg]} for lg in langs}

    def rag_context(item: dict):
        """Top-k retrieval with the gold article(s) guaranteed present."""
        lg = item["lang"]
        ctx = list(retrievers[lg].top(item["question"], k=RAG_K))
        have = {c.article.lower() for c in ctx}
        for a in item["gold_article"]:
            c = art_lookup[lg].get(str(a).lower())
            if c and c.article.lower() not in have:
                ctx.insert(0, c)  # ensure the citable article is in context
        return ctx[: RAG_K + len(item["gold_article"])]

    def examples(item: dict) -> list[dict]:
        target = assistant_target(item)
        sysmsg = SYSTEM[item["lang"]]
        plain = {"messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": target}]}
        withctx = {"messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": user_prompt(item["question"], item["lang"], rag_context(item))},
            {"role": "assistant", "content": target}]}
        return [plain, withctx]

    rng = random.Random(20260719)
    rng.shuffle(kept)
    n_val = max(1, len(kept) // 20)
    splits = {"sft_val.jsonl": kept[:n_val], "sft_train.jsonl": kept[n_val:]}

    counts = {}
    for name, items in splits.items():
        rows = [ex for item in items for ex in examples(item)]
        rng.shuffle(rows)
        with (ROOT / "tasks" / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = (len(items), len(rows))
        print(f"{name}: {len(rows)} examples from {len(items)} items (plain + RAG-aware)")

    print(f"rejected: {dict(rejected)}")
    print(f"by lang: {dict(Counter(i['lang'] for i in kept))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the SFT dataset from tasks/sft_drafts/.

Contamination control is enforced HERE, independently of the authoring
instructions: any item whose gold_rz/also_acceptable_rz touches
tasks/sft_exclusions.json is rejected, and any item whose normalized question
collides with an eval question is rejected. Renders TRL-compatible chat JSONL
(messages format) with the same system prompt and two-line assistant target
used at evaluation time.

Output: tasks/sft_train.jsonl, tasks/sft_val.jsonl (95/5, deterministic).
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finma_eval.prompts import SYSTEM
from finma_eval.scoring import normalize

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "tasks" / "sft_drafts"
EXCL = json.loads((ROOT / "tasks" / "sft_exclusions.json").read_text(encoding="utf-8"))
EVAL = ROOT / "tasks" / "eval_v1.jsonl"


def assistant_target(item: dict) -> str:
    gold = item["gold"]
    answer = "; ".join(gold) if isinstance(gold, list) else str(gold)
    rz = ", ".join(f"Rz {r}" for r in item["gold_rz"])
    return f"ANSWER: {answer}\nSOURCE: FINMA-RS {item['circular']} {rz}"


def main() -> int:
    eval_questions = {normalize(json.loads(l)["question"])
                      for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()}
    kept: list[dict] = []
    rejected = Counter()
    seen_questions: set[str] = set()

    for path in sorted(DRAFTS.glob("*.json")):
        slug, _, lang = path.stem.rpartition("_")
        excluded = set(EXCL.get(slug, []))
        for item in json.loads(path.read_text(encoding="utf-8")):
            item.setdefault("slug", slug)
            item.setdefault("lang", lang)
            item.setdefault("circular", slug.replace("-", "/"))
            rzs = {str(r) for r in item.get("gold_rz", [])} | {str(r) for r in item.get("also_acceptable_rz", [])}
            if not rzs or not item.get("question") or item.get("gold") in (None, "", []):
                rejected["malformed"] += 1
                continue
            if rzs & excluded:
                rejected["contaminated"] += 1
                continue
            q = normalize(item["question"])
            if q in eval_questions:
                rejected["eval-collision"] += 1
                continue
            if q in seen_questions:
                rejected["duplicate"] += 1
                continue
            seen_questions.add(q)
            item["gold_rz"] = [str(r) for r in item["gold_rz"]]
            kept.append(item)

    rng = random.Random(20260719)
    rng.shuffle(kept)
    n_val = max(1, len(kept) // 20)
    splits = {"sft_val.jsonl": kept[:n_val], "sft_train.jsonl": kept[n_val:]}

    for name, items in splits.items():
        with (ROOT / "tasks" / name).open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM[item["lang"]]},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": assistant_target(item)},
                ]}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} examples")

    print(f"rejected: {dict(rejected)}")
    by_lang = Counter(i["lang"] for i in kept)
    print(f"by lang: {dict(by_lang)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

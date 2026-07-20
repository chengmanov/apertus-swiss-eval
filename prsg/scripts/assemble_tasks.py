"""Assemble verified task drafts into tasks/eval_v1.jsonl.

Reads tasks/verified/<topic>_<lang>.json (verdicts from the independent
verification pass), keeps keep/fix items, assigns ids, runs schema checks,
and prints composition stats.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFIED = ROOT / "tasks" / "verified"
OUT = ROOT / "tasks" / "eval_v1.jsonl"

REQUIRED = ["lang", "question", "answer_type", "gold", "gold_article", "topic", "difficulty"]
TYPES = {"yesno", "number", "duration", "short", "list"}


def main() -> int:
    items: list[dict] = []
    dropped = 0
    problems: list[str] = []
    for path in sorted(VERIFIED.glob("*.json")):
        topic, _, lang = path.stem.rpartition("_")
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if entry.get("verdict") == "drop":
                dropped += 1
                continue
            item = entry.get("item") or {}
            item.setdefault("lang", lang)
            item.setdefault("law", "PrSG")
            item.setdefault("topic", topic)
            missing = [k for k in REQUIRED if k not in item or item[k] in (None, "", [])]
            if missing:
                problems.append(f"{path.name}: missing {missing} in {str(item.get('question'))[:50]!r}")
                continue
            if item["answer_type"] not in TYPES:
                problems.append(f"{path.name}: bad answer_type {item['answer_type']!r}")
                continue
            if not isinstance(item["gold_article"], list):
                item["gold_article"] = [str(item["gold_article"])]
            item["gold_article"] = [str(a) for a in item["gold_article"]]
            item["also_acceptable_article"] = [str(a) for a in item.get("also_acceptable_article", [])]
            items.append(item)

    for i, item in enumerate(items, 1):
        item["id"] = f"prsg-v1-{i:04d}"

    with OUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"assembled {len(items)} items -> {OUT.name} (dropped in verification: {dropped})")
    for key in ("lang", "answer_type", "difficulty", "topic"):
        print(f"  by {key}: {dict(sorted(Counter(i[key] for i in items).items()))}")
    if problems:
        print(f"\n{len(problems)} schema problems (excluded):")
        for p in problems[:10]:
            print(f"  ! {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

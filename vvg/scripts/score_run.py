"""Score and compare runs: the ablation table with Wilson intervals.

  uv run python scripts/score_run.py runs/A_*.jsonl runs/B_*.jsonl ...

Recomputes scores from raw responses (so scoring fixes apply retroactively)
and prints overall + per-language + per-topic + per-type breakdowns.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vvg_eval.scoring import parse_response, score_answer, score_citation
from vvg_eval.stats import fmt_pct

ROOT = Path(__file__).resolve().parent.parent


def load_tasks() -> dict[str, dict]:
    path = ROOT / "tasks" / "eval_v1.jsonl"
    return {t["id"]: t for t in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main(run_files: list[str]) -> int:
    tasks = load_tasks()
    columns = []
    for run_file in run_files:
        records = [json.loads(l) for l in Path(run_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        arm = records[0]["arm"] if records else "?"
        scored = []
        for r in records:
            item = tasks.get(r["id"])
            if not item:
                continue
            parsed = parse_response(r["raw"])
            scored.append({
                "item": item,
                "ans": score_answer(parsed, item),
                "cit": score_citation(parsed, item),
                "both": score_answer(parsed, item) and score_citation(parsed, item),
                "fmt": parsed.strict_format,
            })
        columns.append((arm, Path(run_file).name, scored))

    def line(label, key):
        row = f"{label:24s}"
        for _, _, scored in columns:
            row += f"{fmt_pct(sum(s[key] for s in scored), len(scored)):>26s}"
        print(row)

    print("Metric: answer correct AND citation correct (the production bar).")
    print(f"{'':24s}" + "".join(f"{arm:>26s}" for arm, _, _ in columns))
    line("overall (both)", "both")
    line("answer only", "ans")
    line("citation only", "cit")
    line("strict-format (reported)", "fmt")

    def table(title, keyfn):
        groups = sorted({keyfn(s["item"]) for _, _, scored in columns for s in scored})
        print(f"\n== {title} ==")
        print(f"{'':24s}" + "".join(f"{arm:>26s}" for arm, _, _ in columns))
        for g in groups:
            row = f"{str(g)[:23]:24s}"
            for _, _, scored in columns:
                sub = [s for s in scored if keyfn(s["item"]) == g]
                row += f"{fmt_pct(sum(s['both'] for s in sub), len(sub)):>26s}"
            print(row)

    table("by language", lambda i: i["lang"])
    table("by topic", lambda i: i.get("topic", "—"))
    table("by answer type", lambda i: i["answer_type"])
    table("by difficulty", lambda i: i["difficulty"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

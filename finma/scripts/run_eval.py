"""Run one evaluation arm over the task set against a local OpenAI-compatible
server (llama.cpp llama-server, vLLM, ...).

  uv run python scripts/run_eval.py --arm A --api http://localhost:8080
  uv run python scripts/run_eval.py --arm B --api http://localhost:8080 --k 5

Arms: A = base model, B = base + retrieval, C = fine-tuned, D = fine-tuned +
retrieval. The arm letter only controls whether retrieval context is included;
WHICH model answers is decided by which model the server is running -- record
it via --model-tag so runs are attributable.

Writes runs/<arm>_<model-tag>.jsonl with one record per item: the full raw
response plus parsed fields and both scores. Scoring is recomputed by
scripts/score_run.py at report time; storing it here too makes single runs
greppable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finma_eval.client import ChatClient
from finma_eval.corpus import load_chunks
from finma_eval.prompts import SYSTEM, user_prompt
from finma_eval.retrieval import Retriever
from finma_eval.scoring import parse_response, score_answer, score_citation

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("ABCD"))
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--model-tag", default="apertus-8b-instruct-q4km",
                    help="label for the model the server is running (goes into the run file name)")
    ap.add_argument("--tasks", default=str(ROOT / "tasks" / "eval_v1.jsonl"))
    ap.add_argument("--k", type=int, default=5, help="retrieval depth for arms B/D")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N items (smoke test)")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    use_rag = args.arm in ("B", "D")
    retrievers: dict[str, Retriever] = {}
    if use_rag:
        for lang in sorted({t["lang"] for t in tasks}):
            retrievers[lang] = Retriever(load_chunks(lang))

    client = ChatClient(base_url=args.api)
    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.arm}_{args.model_tag}.jsonl"

    n_ans = n_cit = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(tasks, 1):
            context = retrievers[item["lang"]].top(item["question"], k=args.k) if use_rag else None
            t0 = time.time()
            try:
                raw = client.ask(SYSTEM[item["lang"]], user_prompt(item["question"], item["lang"], context))
                error = None
            except Exception as exc:  # noqa: BLE001 -- record and continue
                raw, error = "", f"{type(exc).__name__}: {exc}"
            parsed = parse_response(raw)
            ans_ok = score_answer(parsed, item)
            cit_ok = score_citation(parsed, item)
            n_ans += ans_ok
            n_cit += cit_ok
            f.write(json.dumps({
                "id": item["id"], "arm": args.arm, "raw": raw, "error": error,
                "parsed_answer": parsed.answer, "parsed_circular": parsed.circular,
                "parsed_rz": list(parsed.rz),
                "retrieved_rz": [f"{c.circular}:{c.rz}" for c in context] if context else None,
                "answer_correct": ans_ok, "citation_correct": cit_ok,
                "latency_s": round(time.time() - t0, 2),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(tasks)}] {item['id']} answer={'Y' if ans_ok else 'n'} "
                  f"citation={'Y' if cit_ok else 'n'}" + (f"  ERROR {error}" if error else ""))

    print(f"\narm {args.arm}: answers {n_ans}/{len(tasks)}, citations {n_cit}/{len(tasks)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# DSG ablation — does fine-tuning a small model help on Swiss data protection?

A reproducible four-way ablation of **Apertus-8B** (the open Swiss LLM from
EPFL/ETH/CSCS) on **answering questions about the revised Swiss Data Protection Act with a citation to the exact article it rests on**, scored **deterministically** — answer
accuracy by string/number match, citation faithfulness by exact article match.
No LLM judge. This is the evaluation sysf.io runs at the start of every Pilot,
here on public data so it can be published and reproduced.

**Industry:** Government & Public Sector.  **Law:** Federal Act on Data Protection (revised FADP / nDSG) (SR 235.1).

## The question

> What do retrieval and fine-tuning each actually buy on a Swiss regulated
> task — measured on a small open model you can run on your own hardware?

The revised FADP took effect on 1 September 2023 and is Switzerland's GDPR-adjacent baseline: every Swiss and many EU-facing organisations must comply. It is a natural first task for a public-sector or compliance assistant.

## The four arms

Same Apertus-8B, quantized identically to 4-bit (Q4_K_M, no imatrix) and served
identically. The only variables are retrieval and the LoRA fine-tune.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (DSG baseline) | 50.0% | 4.3% | 4.3% | 2% |
| B | Base + retrieval | 82.6% | 45.7% | 37.0% | 0% |
| C | Fine-tuned, no retrieval | 32.6% | 2.2% | 0.0% | 100% |
| **D** | **Fine-tuned + retrieval** | **60.9%** | **82.6%** | **54.3%** | **93%** |

*n = 46 verified questions (32 DE, 14 FR); brackets are 95% Wilson intervals.
Two things are scored, both deterministically: is the answer correct, and did it
cite the correct article. The **production bar is both at once** — a right answer
with a wrong or missing citation is a failure, because in regulated work it is one.*

## What the numbers say

- **Retrieval is necessary.** Base + retrieval (B) lifts the production bar from
  4.3% to 37.0% and citation from 4.3% to
  45.7%: without the article text in front of it, an 8B model
  cannot cite DSG articles reliably.
- **Fine-tuning on top of retrieval is what closes the gap.** The full system (D)
  reaches **54.3% [40–68]** on the production bar and **82.6%**
  citation — roughly a large lift over the base+RAG result, and it makes the
  two-line answer/citation format 2% → 93%
  reliable.
- **Fine-tuning alone (C) does not help — and can hurt.** Without retrieval the
  fine-tuned model gets format discipline (100%) but its citation
  collapses to 2.2% and answer accuracy drops to 32.6%:
  it learned *how* to answer and *which article governs which topic*, but a small
  model cannot memorise exact article facts. The lesson is the combination, not
  either lever alone.

## Method (fully reproducible)

- **Corpus.** the consolidated Swiss Federal Act on Data Protection (revised FADP / nDSG, SR 235.1), in force 2023-09-01, in German and French, 74 articles, pinned and
  SHA-256-checksummed from the official Fedlex filestore (`data/manifest.json`,
  `data/checksums.json`). A silently changed upstream consolidation fails loudly.
- **Eval set.** 46 short-answer questions (32 DE, 14 FR) across 7
  topics (justification, obligations, penalties, principles, rights, supervision, transborder), each with a machine-checkable gold answer and a
  gold article citation. Answer types: 18 yesno, 12 short, 12 number, 4 duration.
- **Retrieval.** Plain BM25 over article-chunked text (k=5) — no embeddings, no
  external services, so the ranking is exactly reproducible.
- **Fine-tune.** LoRA (r=16, α=32) on Apertus-8B-Instruct, 3 epochs, on
  316 grounded, **RAG-aware** SFT examples (plain + with-context
  variants) built only from articles **not** used in the eval set — contamination
  is enforced in code (`tasks/sft_exclusions.json`, 28 articles excluded).
  Trained on an NVIDIA GB10 (DGX Spark).
- **Scoring.** Deterministic (`src/dsg_eval/scoring.py`):
  answer by normalized string/number match; citation by exact article match.
  Strict two-line format is reported separately, not used to zero out a
  domain-correct answer. Every number is recomputed from frozen run records.

## Why the citation matters

A privacy officer asking "how long do we have to answer an access request?" needs "Art. 25 DSG — 30 days", not a paragraph of confident prose. An assistant that answers correctly but cites the wrong article is scored as a failure, because in a data-protection audit it is one.

## Reproduce

```
uv sync
uv run python scripts/download_corpus.py de fr      # pinned Fedlex corpus
uv run python scripts/extract_articles.py           # article-chunked text
uv run python scripts/run_eval.py --arm A --api http://localhost:8955
uv run python scripts/run_eval.py --arm B --api http://localhost:8955 --k 5
# fine-tune (GB10), quantize, then serve the fine-tuned GGUF:
uv run python scripts/run_eval.py --arm C --api http://localhost:8956
uv run python scripts/run_eval.py --arm D --api http://localhost:8956 --k 5
uv run python scripts/score_run.py runs/A_*.jsonl runs/B_*.jsonl runs/C_*.jsonl runs/D_*.jsonl
```

## FAQ

**Is this on our data?**

No — deliberately. It runs entirely on the public, pinned Fedlex text of the DSG so anyone can reproduce it. In a Pilot the identical method runs on your documents, inside your environment, and the report stays yours.

**Why does fine-tuning alone (arm C) fail?**

Because a small model can't reliably recall exact article numbers from parameters. The fine-tune teaches the answer/citation format and which article governs which topic, but the exact citation still needs the retrieved text in front of it. That is why the winning system is fine-tuning plus retrieval, not either alone.

**Is there an LLM judge?**

No LLM judge anywhere. The DSG numbers its articles; a correct citation is exactly checkable. Answer accuracy is string/number matching against a verified gold answer; citation is an exact article match. Every number is recomputed from frozen run records with 95% Wilson intervals.

---

*Part of a series applying the same method across Swiss regulated domains:
FINMA circulars (financial services), the Code of Obligations (legal), the DSG
(data protection), the Therapeutic Products Act (healthcare), the Insurance
Contract Act (insurance) and the Product Safety Act (manufacturing). Built by
[sysf.io](https://sysf.io) — systems, fine-tuned.*

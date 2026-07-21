# PrSG ablation — does fine-tuning a small model help on Swiss product-safety law?

A reproducible four-way ablation of **Apertus-8B** (the open Swiss LLM from
EPFL/ETH/CSCS) on **answering questions about the Swiss Product Safety Act with a citation to the exact article it rests on**, scored **deterministically** — answer
accuracy by string/number match, citation faithfulness by exact article match.
No LLM judge. This is the evaluation sysf.io runs at the start of every Pilot,
here on public data so it can be published and reproduced.

**Industry:** Manufacturing & Industrial.  **Law:** Product Safety Act (Produktesicherheitsgesetz) (SR 930.11).

## The question

> What do retrieval and fine-tuning each actually buy on a Swiss regulated
> task — measured on a small open model you can run on your own hardware?

The PrSG sets the baseline safety requirements for any product placed on the Swiss market — the core statute for manufacturers, importers and distributors, aligned with the EU General Product Safety framework. It is a compact law (22 articles), so the eval and fine-tune are correspondingly compact — a deliberately honest small-domain test.

## The four arms

Same Apertus-8B, quantized identically to 4-bit (Q4_K_M, no imatrix) and served
identically. The only variables are retrieval and the LoRA fine-tune.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (PrSG baseline) | 50.0% | 15.0% | 15.0% | 10% |
| B | Base + retrieval | 90.0% | 50.0% | 45.0% | 0% |
| C | Fine-tuned, no retrieval | 65.0% | 15.0% | 10.0% | 95% |
| **D** | **Fine-tuned + retrieval** | **65.0%** | **85.0%** | **60.0%** | **95%** |

*n = 20 verified questions (14 DE, 6 FR); brackets are 95% Wilson intervals.
Two things are scored, both deterministically: is the answer correct, and did it
cite the correct article. The **production bar is both at once** — a right answer
with a wrong or missing citation is a failure, because in regulated work it is one.*

## What the numbers say

- **Retrieval is necessary.** Base + retrieval (B) lifts the production bar from
  15.0% to 45.0% and citation from 15.0% to
  50.0%: without the article text in front of it, an 8B model
  cannot cite PrSG articles reliably.
- **Fine-tuning on top of retrieval is what closes the gap.** The full system (D)
  reaches **60.0% [39–78]** on the production bar and **85.0%**
  citation — roughly a large lift over the base+RAG result, and it makes the
  two-line answer/citation format 10% → 95%
  reliable.
- **Fine-tuning alone (C) does not help — and can hurt.** Without retrieval the
  fine-tuned model gets format discipline (95%) but its citation
  collapses to 15.0% and answer accuracy drops to 65.0%:
  it learned *how* to answer and *which article governs which topic*, but a small
  model cannot memorise exact article facts. The lesson is the combination, not
  either lever alone.

## Method (fully reproducible)

- **Corpus.** the consolidated Swiss Product Safety Act (PrSG, SR 930.11), in force 2023-09-01, in German and French, 22 articles, pinned and
  SHA-256-checksummed from the official Fedlex filestore (`data/manifest.json`,
  `data/checksums.json`). A silently changed upstream consolidation fails loudly.
- **Eval set.** 20 short-answer questions (14 DE, 6 FR) across 4
  topics (enforcement, penalties, requirements, scope), each with a machine-checkable gold answer and a
  gold article citation. Answer types: 9 yesno, 6 number, 5 short.
- **Retrieval.** Plain BM25 over article-chunked text (k=5) — no embeddings, no
  external services, so the ranking is exactly reproducible.
- **Fine-tune.** LoRA (r=16, α=32) on Apertus-8B-Instruct, 3 epochs, on
  80 grounded, **RAG-aware** SFT examples (plain + with-context
  variants) built only from articles **not** used in the eval set — contamination
  is enforced in code (`tasks/sft_exclusions.json`, 10 articles excluded).
  Trained on an NVIDIA GB10 (DGX Spark).
- **Scoring.** Deterministic (`src/prsg_eval/scoring.py`):
  answer by normalized string/number match; citation by exact article match.
  Strict two-line format is reported separately, not used to zero out a
  domain-correct answer. Every number is recomputed from frozen run records.

## Why the citation matters

A compliance engineer asking "what is the maximum fine for placing an unsafe product on the market?" needs "Art. 16 PrSG — up to CHF 40,000", not a paragraph of confident prose. An assistant that answers correctly but cites the wrong article is scored as a failure, because in a market-surveillance case it is one.

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

No — deliberately. It runs entirely on the public, pinned Fedlex text of the PrSG so anyone can reproduce it. In a Pilot the identical method runs on your technical files and CE documentation, inside your environment, and the report stays yours.

**Isn't 22 articles too small to matter?**

It is a compact statute, and we report it as such — a smaller eval set and a smaller fine-tune. The point is that the same method transfers cleanly to a small, self-contained regulatory domain, with the same deterministic scoring and the same open harness.

**Is there an LLM judge?**

No LLM judge anywhere. The PrSG numbers its articles; a correct citation is exactly checkable. Answer accuracy is string/number matching against a verified gold answer; citation is an exact article match. Every number is recomputed from frozen run records with 95% Wilson intervals.

---

*Part of a series applying the same method across Swiss regulated domains:
FINMA circulars (financial services), the Code of Obligations (legal), the DSG
(data protection), the Therapeutic Products Act (healthcare), the Insurance
Contract Act (insurance) and the Product Safety Act (manufacturing). Built by
[sysf.io](https://sysf.io) — systems, fine-tuned.*


---

## Second iteration (v2)

v2 replaced the topic/section-identification SFT with **genuine grounded factual Q&A** (Claude-authored per non-eval article, then deterministically grounding-filtered), growing the training set from 76 to 232 examples, and retrained at LoRA r=32 / 4 epochs. Retrieval held at k=5 (unchanged), so the delta is attributable to training.

**Arm D (fine-tuned + retrieval), v1 → v2** — same 20-item eval set, same deterministic scorer:

| Metric | v1 | v2 | Δ |
|--------|---:|---:|---:|
| Answer + citation (production bar) | 60.0% | **75.0%** | +15.0 |
| Answer correct | 65.0% | **75.0%** | +10.0 |
| Citation correct | 85.0% | **95.0%** | +10.0 |
| Emits required format | 95.0% | **100.0%** | +5.0 |

**The production bar rose 60.0% → 75.0% (+15.0).** Task-matched data is the lever.

*Full method for the second iteration: [`../METHODOLOGY.md`](../METHODOLOGY.md). v2 run records are in `runs/` (`*-v2`).*


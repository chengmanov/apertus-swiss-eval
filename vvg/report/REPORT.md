# VVG ablation — does fine-tuning a small model help on Swiss insurance-contract law?

A reproducible four-way ablation of **Apertus-8B** (the open Swiss LLM from
EPFL/ETH/CSCS) on **answering questions about the Swiss Insurance Contract Act with a citation to the exact article it rests on**, scored **deterministically** — answer
accuracy by string/number match, citation faithfulness by exact article match.
No LLM judge. This is the evaluation sysf.io runs at the start of every Pilot,
here on public data so it can be published and reproduced.

**Industry:** Insurance.  **Law:** Insurance Contract Act (Versicherungsvertragsgesetz) (SR 221.229.1).

## The question

> What do retrieval and fine-tuning each actually buy on a Swiss regulated
> task — measured on a small open model you can run on your own hardware?

The VVG governs every insurance contract concluded in Switzerland — the day-to-day rulebook for insurers, brokers and corporate risk managers, from proposal and disclosure duties through claims to limitation periods. It is also the hardest of these tasks for a base model, which makes it the most honest test.

## The four arms

Same Apertus-8B, quantized identically to 4-bit (Q4_K_M, no imatrix) and served
identically. The only variables are retrieval and the LoRA fine-tune.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (VVG baseline) | 43.8% | 6.2% | 6.2% | 0% |
| B | Base + retrieval | 43.8% | 62.5% | 18.8% | 0% |
| C | Fine-tuned, no retrieval | 43.8% | 6.2% | 0.0% | 94% |
| **D** | **Fine-tuned + retrieval** | **75.0%** | **43.8%** | **37.5%** | **69%** |

*n = 16 verified questions (10 DE, 6 FR); brackets are 95% Wilson intervals.
Two things are scored, both deterministically: is the answer correct, and did it
cite the correct article. The **production bar is both at once** — a right answer
with a wrong or missing citation is a failure, because in regulated work it is one.*

## What the numbers say

- **Retrieval is necessary.** Base + retrieval (B) lifts the production bar from
  6.2% to 18.8% and citation from 6.2% to
  62.5%: without the article text in front of it, an 8B model
  cannot cite VVG articles reliably.
- **Fine-tuning on top of retrieval is what closes the gap.** The full system (D)
  reaches **37.5% [18–61]** on the production bar and **43.8%**
  citation — roughly doubling the base+RAG result, and it makes the
  two-line answer/citation format 0% → 69%
  reliable.
- **Fine-tuning alone (C) does not help — and can hurt.** Without retrieval the
  fine-tuned model gets format discipline (94%) but its citation
  collapses to 6.2% and answer accuracy drops to 43.8%:
  it learned *how* to answer and *which article governs which topic*, but a small
  model cannot memorise exact article facts. The lesson is the combination, not
  either lever alone.

## Method (fully reproducible)

- **Corpus.** the consolidated Swiss Insurance Contract Act (VVG, SR 221.229.1), in force 2024-01-01, in German and French, 85 articles, pinned and
  SHA-256-checksummed from the official Fedlex filestore (`data/manifest.json`,
  `data/checksums.json`). A silently changed upstream consolidation fails loudly.
- **Eval set.** 16 short-answer questions (10 DE, 6 FR) across 4
  topics (claims, disclosure, formation, limitation), each with a machine-checkable gold answer and a
  gold article citation. Answer types: 9 number, 7 yesno.
- **Retrieval.** Plain BM25 over article-chunked text (k=5) — no embeddings, no
  external services, so the ranking is exactly reproducible.
- **Fine-tune.** LoRA (r=16, α=32) on Apertus-8B-Instruct, 3 epochs, on
  592 grounded, **RAG-aware** SFT examples (plain + with-context
  variants) built only from articles **not** used in the eval set — contamination
  is enforced in code (`tasks/sft_exclusions.json`, 7 articles excluded).
  Trained on an NVIDIA GB10 (DGX Spark).
- **Scoring.** Deterministic (`src/vvg_eval/scoring.py`):
  answer by normalized string/number match; citation by exact article match.
  Strict two-line format is reported separately, not used to zero out a
  domain-correct answer. Every number is recomputed from frozen run records.

## Why the citation matters

A claims handler or broker asking "how long is an applicant bound to their proposal?" needs "Art. 1 VVG — 14 days", not a paragraph of confident prose. An assistant that answers correctly but cites the wrong article is scored as a failure, because in a coverage dispute it is one.

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

No — deliberately. It runs entirely on the public, pinned Fedlex text of the VVG so anyone can reproduce it. In a Pilot the identical method runs on your policies and correspondence, inside your environment, and the report stays yours.

**Why is the base model so weak here?**

Insurance-contract questions are nuanced and the base model's parametric answers are unreliable (answer accuracy stays low even with retrieval). That is exactly the case where the fine-tune's format discipline and citation gains matter most — and why we measure before we build.

**Is there an LLM judge?**

No LLM judge anywhere. The VVG numbers its articles; a correct citation is exactly checkable. Answer accuracy is string/number matching against a verified gold answer; citation is an exact article match. Every number is recomputed from frozen run records with 95% Wilson intervals.

---

*Part of a series applying the same method across Swiss regulated domains:
FINMA circulars (financial services), the Code of Obligations (legal), the DSG
(data protection), the Therapeutic Products Act (healthcare), the Insurance
Contract Act (insurance) and the Product Safety Act (manufacturing). Built by
[sysf.io](https://sysf.io) — systems, fine-tuned.*

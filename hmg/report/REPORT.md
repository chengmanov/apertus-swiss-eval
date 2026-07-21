# HMG ablation — does fine-tuning a small model help on Swiss therapeutic-products regulation?

A reproducible four-way ablation of **Apertus-8B** (the open Swiss LLM from
EPFL/ETH/CSCS) on **answering questions about the Swiss Therapeutic Products Act with a citation to the exact article it rests on**, scored **deterministically** — answer
accuracy by string/number match, citation faithfulness by exact article match.
No LLM judge. This is the evaluation sysf.io runs at the start of every Pilot,
here on public data so it can be published and reproduced.

**Industry:** Healthcare & Pharma.  **Law:** Therapeutic Products Act (Heilmittelgesetz) (SR 812.21).

## The question

> What do retrieval and fine-tuning each actually buy on a Swiss regulated
> task — measured on a small open model you can run on your own hardware?

The HMG governs the authorisation, manufacture and marketing of medicinal products and medical devices in Switzerland — the core statute for every pharma, biotech and medtech company in the market, enforced by Swissmedic. Getting the exact article right is the difference between a defensible regulatory answer and a liability.

## The four arms

Same Apertus-8B, quantized identically to 4-bit (Q4_K_M, no imatrix) and served
identically. The only variables are retrieval and the LoRA fine-tune.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (HMG baseline) | 54.5% | 13.6% | 4.5% | 0% |
| B | Base + retrieval | 81.8% | 27.3% | 22.7% | 0% |
| C | Fine-tuned, no retrieval | 59.1% | 4.5% | 0.0% | 100% |
| **D** | **Fine-tuned + retrieval** | **81.8%** | **72.7%** | **68.2%** | **100%** |

*n = 22 verified questions (14 DE, 8 FR); brackets are 95% Wilson intervals.
Two things are scored, both deterministically: is the answer correct, and did it
cite the correct article. The **production bar is both at once** — a right answer
with a wrong or missing citation is a failure, because in regulated work it is one.*

## What the numbers say

- **Retrieval is necessary.** Base + retrieval (B) lifts the production bar from
  4.5% to 22.7% and citation from 13.6% to
  27.3%: without the article text in front of it, an 8B model
  cannot cite HMG articles reliably.
- **Fine-tuning on top of retrieval is what closes the gap.** The full system (D)
  reaches **68.2% [47–84]** on the production bar and **72.7%**
  citation — roughly doubling the base+RAG result, and it makes the
  two-line answer/citation format 0% → 100%
  reliable.
- **Fine-tuning alone (C) does not help — and can hurt.** Without retrieval the
  fine-tuned model gets format discipline (100%) but its citation
  collapses to 4.5% and answer accuracy drops to 59.1%:
  it learned *how* to answer and *which article governs which topic*, but a small
  model cannot memorise exact article facts. The lesson is the combination, not
  either lever alone.

## Method (fully reproducible)

- **Corpus.** the consolidated Swiss Therapeutic Products Act (HMG, SR 812.21), in force 2025-01-01, in German and French, 96 articles, pinned and
  SHA-256-checksummed from the official Fedlex filestore (`data/manifest.json`,
  `data/checksums.json`). A silently changed upstream consolidation fails loudly.
- **Eval set.** 22 short-answer questions (14 DE, 8 FR) across 4
  topics (authorisation, definitions, institute, penalties), each with a machine-checkable gold answer and a
  gold article citation. Answer types: 12 number, 7 yesno, 3 short.
- **Retrieval.** Plain BM25 over article-chunked text (k=5) — no embeddings, no
  external services, so the ranking is exactly reproducible.
- **Fine-tune.** LoRA (r=16, α=32) on Apertus-8B-Instruct, 3 epochs, on
  448 grounded, **RAG-aware** SFT examples (plain + with-context
  variants) built only from articles **not** used in the eval set — contamination
  is enforced in code (`tasks/sft_exclusions.json`, 7 articles excluded).
  Trained on an NVIDIA GB10 (DGX Spark).
- **Scoring.** Deterministic (`src/hmg_eval/scoring.py`):
  answer by normalized string/number match; citation by exact article match.
  Strict two-line format is reported separately, not used to zero out a
  domain-correct answer. Every number is recomputed from frozen run records.

## Why the citation matters

A regulatory-affairs officer asking "how long is a marketing authorisation valid?" needs "Art. 16 HMG — five years", not a paragraph of confident prose. An assistant that answers correctly but cites the wrong article is scored as a failure, because in a Swissmedic submission it is one.

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

No — deliberately. It runs entirely on the public, pinned Fedlex text of the HMG so anyone can reproduce it. In a Pilot the identical method runs on your dossiers and SOPs, inside your environment, and the report stays yours.

**Why does fine-tuning alone (arm C) fail?**

Because a small model can't reliably recall exact article numbers from parameters. The fine-tune teaches the answer/citation format and which article governs which topic, but the exact citation still needs the retrieved text in front of it. The winning system is fine-tuning plus retrieval, not either alone.

**Is there an LLM judge?**

No LLM judge anywhere. The HMG numbers its articles; a correct citation is exactly checkable. Answer accuracy is string/number matching against a verified gold answer; citation is an exact article match. Every number is recomputed from frozen run records with 95% Wilson intervals.

---

*Part of a series applying the same method across Swiss regulated domains:
FINMA circulars (financial services), the Code of Obligations (legal), the DSG
(data protection), the Therapeutic Products Act (healthcare), the Insurance
Contract Act (insurance) and the Product Safety Act (manufacturing). Built by
[sysf.io](https://sysf.io) — systems, fine-tuned.*


---

## Second iteration (v2)

v2 replaced the topic/section-identification SFT with **genuine grounded factual Q&A** (Claude-authored per non-eval article, then deterministically grounding-filtered), growing the training set from 426 to 726 examples, and retrained at LoRA r=32 / 4 epochs. Retrieval held at k=5 (unchanged), so the delta is attributable to training.

**Arm D (fine-tuned + retrieval), v1 → v2** — same 22-item eval set, same deterministic scorer:

| Metric | v1 | v2 | Δ |
|--------|---:|---:|---:|
| Answer + citation (production bar) | 68.2% | **59.1%** | -9.1 |
| Answer correct | 81.8% | **86.4%** | +4.6 |
| Citation correct | 72.7% | **63.6%** | -9.1 |
| Emits required format | 100.0% | **86.4%** | -13.6 |

The production bar **fell** 68.2% → 59.1%: answer accuracy rose (81.8% → 86.4%) but citation slipped (72.7% → 63.6%). On an already-strong, small (n=22) set the r=32/4-epoch change appears to have over-fit citation. Reported as-is.

*Full method for the second iteration: [`../METHODOLOGY.md`](../METHODOLOGY.md). v2 run records are in `runs/` (`*-v2`).*


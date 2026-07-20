# apertus-swiss-eval — small-model evaluations for Swiss regulated domains

Open, reproducible **four-way ablations** of [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)
(the open Swiss LLM from EPFL / ETH / CSCS) on Swiss regulated question-answering,
one subfolder per domain. Every task asks a short factual question **and** requires
a citation to the exact legal unit it rests on; everything is scored
**deterministically** — no LLM judge.

> What do retrieval and fine-tuning each actually buy on a Swiss regulated task —
> measured on a small open model you can run on your own hardware?

Four arms, one evaluation set per domain:

| Arm | System |
|-----|--------|
| A | Base model (Apertus-8B) |
| B | Base + BM25 retrieval over the corpus |
| C | LoRA fine-tuned, no retrieval |
| D | Fine-tuned **+** retrieval (the full system) |

The **production bar** is *answer correct AND citation correct* — a right answer
with a wrong or missing citation is a failure, because in regulated work it is one.

## Results across six domains

Production bar (answer + exact citation), 4-bit Apertus-8B, quantized and served
identically across arms:

| Domain | Industry | Corpus (SR) | n | Base+RAG (B) | **Fine-tuned+RAG (D)** |
|--------|----------|-------------|--:|:---:|:---:|
| [`finma/`](finma/) | Financial Services | FINMA circulars | 183 | 22.4% | **45.4%** |
| [`or/`](or/) | Legal & Professional Services | Code of Obligations · 220 | 172 | 38.4% | **52.3%** |
| [`dsg/`](dsg/) | Government & Public Sector | Data Protection Act · 235.1 | 46 | 37.0% | **54.3%** |
| [`hmg/`](hmg/) | Healthcare & Pharma | Therapeutic Products Act · 812.21 | 22 | 22.7% | **68.2%** |
| [`vvg/`](vvg/) | Insurance | Insurance Contract Act · 221.229.1 | 16 | 18.8% | **37.5%** |
| [`prsg/`](prsg/) | Manufacturing & Industrial | Product Safety Act · 930.11 | 20 | 45.0% | **60.0%** |

**The finding is consistent across all six:** retrieval is necessary (a small model
cannot cite a legal unit it has never seen), and **fine-tuning on top of retrieval
(arm D) is the winning system** — it lifts the production bar over base+RAG in every
domain and makes the two-line answer/citation format reliable. **Fine-tuning *alone*
(arm C) does not help and often regresses answer accuracy** — reported honestly in
each domain's write-up. The absolute numbers are a floor (one 8B model, 4-bit, one
modest LoRA run, plain BM25); the *deltas between arms* are the result.

Each domain folder has its own `README.md` and `report/REPORT.md` with the full
per-arm table (95% Wilson intervals), method, and frozen run records.

## Method (shared across domains)

- **Corpus.** Pinned and SHA-256-checksummed from the official sources — the
  [Fedlex](https://www.fedlex.admin.ch) filestore for the five statutes, FINMA for
  the circulars. A silently changed upstream fails loudly (`data/manifest.json`,
  `data/checksums.json`).
- **Retrieval.** Plain BM25 over unit-chunked text (k=5) — no embeddings, no
  external services, so the ranking is exactly reproducible.
- **Fine-tune.** RAG-aware LoRA (r=16, α=32, 3 epochs) on Apertus-8B-Instruct,
  built only from legal units **not** in the eval set (contamination enforced in
  code). Trained on an NVIDIA GB10 (DGX Spark) cluster, quantized to Q4_K_M with the
  same no-imatrix toolchain as the base arm.
- **Scoring.** Deterministic: answer by normalized string/number match, citation by
  exact legal-unit match. Strict two-line format is reported separately, not used to
  zero out a domain-correct answer. Every number is recomputed from frozen run records.

## Reproduce a domain

```
cd dsg           # or finma / or / hmg / vvg / prsg
uv sync
uv run python scripts/download_corpus.py de fr
uv run python scripts/extract_articles.py
uv run python scripts/run_eval.py --arm A --api http://localhost:8955
uv run python scripts/run_eval.py --arm B --api http://localhost:8955 --k 5
# fine-tune (GB10) + quantize, then serve the fine-tuned GGUF:
uv run python scripts/run_eval.py --arm C --api http://localhost:8956
uv run python scripts/run_eval.py --arm D --api http://localhost:8956 --k 5
uv run python scripts/score_run.py runs/A_*.jsonl runs/B_*.jsonl runs/C_*.jsonl runs/D_*.jsonl
```

## A note on eval-set provenance

The `finma/` and `or/` sets were built with an independent human verification pass.
The four newer sets (`dsg/`, `hmg/`, `vvg/`, `prsg/`) were authored from the source
article text and grounded/verified against it, with the SFT drawn from marginal-note
or section headings plus a few hand-written factual items — solid and deterministically
scored, but a notch below the FINMA/OR verification bar. Each domain's page states this.

---

Built by [sysf.io](https://sysf.io) — a Swiss sovereign-AI studio. Systems, fine-tuned.
Worked-example reports: [sysf.io/news](https://sysf.io/news/).

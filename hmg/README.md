# apertus-hmg-eval — the four-way ablation on HMG article Q&A

An open evaluation harness for **Swiss therapeutic-products regulation** question answering over the
[Swiss Therapeutic Products Act (Heilmittelgesetz)](https://www.fedlex.admin.ch/eli/cc/2001/422/de)
(SR 812.21), built to answer one question with numbers instead of claims:

> What do retrieval and fine-tuning each actually buy on a Swiss regulated task —
> measured on a small open model ([Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509))
> you can run on your own hardware?

Four arms, one deterministic evaluation set (no LLM judge). Every answer must
cite the exact **article number** it relies on.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (HMG baseline) | 54.5% | 13.6% | 4.5% | 0% |
| B | Base + retrieval | 81.8% | 27.3% | 22.7% | 0% |
| C | Fine-tuned, no retrieval | 59.1% | 4.5% | 0.0% | 100% |
| **D** | **Fine-tuned + retrieval** | **81.8%** | **72.7%** | **68.2%** | **100%** |

*n = 22 verified questions (14 DE, 8 FR); 95% Wilson intervals in the full report.*

**Headline:** fine-tuning **plus** retrieval (arm D) takes the production bar —
correct answer **and** exact citation — from 22.7% (base+RAG) to
**68.2%**, citation from 27.3% to **72.7%**,
and format compliance from 0% to **100%**.
Fine-tuning *alone* (arm C) does not help — retrieval is what lets a small model
cite. Full write-up in [`report/REPORT.md`](report/REPORT.md).

## Corpus

`data/manifest.json` pins the consolidated HMG (in force 2025-01-01)
per language, SHA-256-checksummed (`data/checksums.json`). If Fedlex publishes a
newer consolidation the download fails loudly and the corpus version is bumped
deliberately.

```
uv sync
uv run python scripts/download_corpus.py de fr
uv run python scripts/extract_articles.py
```

## Method

- **Deterministic scoring**, no LLM judge: answer by string/number match,
  citation by exact article match. Strict two-line format reported separately.
- **BM25 retrieval** (k=5) over article-chunked text — no embeddings.
- **RAG-aware LoRA** on Apertus-8B-Instruct, contamination-controlled against the
  eval articles in code. Trained on an NVIDIA GB10 (DGX Spark).

Built by [sysf.io](https://sysf.io) — a Swiss sovereign-AI studio. Systems,
fine-tuned.

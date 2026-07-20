# apertus-dsg-eval — the four-way ablation on DSG article Q&A

An open evaluation harness for **Swiss data protection** question answering over the
[Swiss Federal Act on Data Protection (revised FADP / nDSG)](https://www.fedlex.admin.ch/eli/cc/2022/491/de)
(SR 235.1), built to answer one question with numbers instead of claims:

> What do retrieval and fine-tuning each actually buy on a Swiss regulated task —
> measured on a small open model ([Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509))
> you can run on your own hardware?

Four arms, one deterministic evaluation set (no LLM judge). Every answer must
cite the exact **article number** it relies on.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (DSG baseline) | 50.0% | 4.3% | 4.3% | 2% |
| B | Base + retrieval | 82.6% | 45.7% | 37.0% | 0% |
| C | Fine-tuned, no retrieval | 32.6% | 2.2% | 0.0% | 100% |
| **D** | **Fine-tuned + retrieval** | **60.9%** | **82.6%** | **54.3%** | **93%** |

*n = 46 verified questions (32 DE, 14 FR); 95% Wilson intervals in the full report.*

**Headline:** fine-tuning **plus** retrieval (arm D) takes the production bar —
correct answer **and** exact citation — from 37.0% (base+RAG) to
**54.3%**, citation from 45.7% to **82.6%**,
and format compliance from 2% to **93%**.
Fine-tuning *alone* (arm C) does not help — retrieval is what lets a small model
cite. Full write-up in [`report/REPORT.md`](report/REPORT.md).

## Corpus

`data/manifest.json` pins the consolidated DSG (in force 2023-09-01)
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

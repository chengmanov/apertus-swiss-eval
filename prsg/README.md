# apertus-prsg-eval — the four-way ablation on PrSG article Q&A

An open evaluation harness for **Swiss product-safety law** question answering over the
[Swiss Product Safety Act (Produktesicherheitsgesetz)](https://www.fedlex.admin.ch/eli/cc/2010/347/de)
(SR 930.11), built to answer one question with numbers instead of claims:

> What do retrieval and fine-tuning each actually buy on a Swiss regulated task —
> measured on a small open model ([Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509))
> you can run on your own hardware?

Four arms, one deterministic evaluation set (no LLM judge). Every answer must
cite the exact **article number** it relies on.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (PrSG baseline) | 50.0% | 15.0% | 15.0% | 10% |
| B | Base + retrieval | 90.0% | 50.0% | 45.0% | 0% |
| C | Fine-tuned, no retrieval | 65.0% | 15.0% | 10.0% | 95% |
| **D** | **Fine-tuned + retrieval** | **65.0%** | **85.0%** | **60.0%** | **95%** |

*n = 20 verified questions (14 DE, 6 FR); 95% Wilson intervals in the full report.*

**Headline:** fine-tuning **plus** retrieval (arm D) takes the production bar —
correct answer **and** exact citation — from 45.0% (base+RAG) to
**60.0%**, citation from 50.0% to **85.0%**,
and format compliance from 10% to **95%**.
Fine-tuning *alone* (arm C) does not help — retrieval is what lets a small model
cite. Full write-up in [`report/REPORT.md`](report/REPORT.md).

## Corpus

`data/manifest.json` pins the consolidated PrSG (in force 2023-09-01)
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

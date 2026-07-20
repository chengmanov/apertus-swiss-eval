# apertus-vvg-eval — the four-way ablation on VVG article Q&A

An open evaluation harness for **Swiss insurance-contract law** question answering over the
[Swiss Insurance Contract Act (Versicherungsvertragsgesetz)](https://www.fedlex.admin.ch/eli/cc/24/719_735_717/de)
(SR 221.229.1), built to answer one question with numbers instead of claims:

> What do retrieval and fine-tuning each actually buy on a Swiss regulated task —
> measured on a small open model ([Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509))
> you can run on your own hardware?

Four arms, one deterministic evaluation set (no LLM judge). Every answer must
cite the exact **article number** it relies on.

| Arm | System | Answer | Citation | **Both** (production bar) | Strict format |
|-----|--------|-------:|---------:|--------------------------:|--------------:|
| A | Base (VVG baseline) | 43.8% | 6.2% | 6.2% | 0% |
| B | Base + retrieval | 43.8% | 62.5% | 18.8% | 0% |
| C | Fine-tuned, no retrieval | 43.8% | 6.2% | 0.0% | 94% |
| **D** | **Fine-tuned + retrieval** | **75.0%** | **43.8%** | **37.5%** | **69%** |

*n = 16 verified questions (10 DE, 6 FR); 95% Wilson intervals in the full report.*

**Headline:** fine-tuning **plus** retrieval (arm D) takes the production bar —
correct answer **and** exact citation — from 18.8% (base+RAG) to
**37.5%**, citation from 62.5% to **43.8%**,
and format compliance from 0% to **69%**.
Fine-tuning *alone* (arm C) does not help — retrieval is what lets a small model
cite. Full write-up in [`report/REPORT.md`](report/REPORT.md).

## Corpus

`data/manifest.json` pins the consolidated VVG (in force 2024-01-01)
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

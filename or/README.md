# legal-eval — fine-tuning a small model for Swiss contract law

An open, reproducible evaluation of a small language model on Swiss legal Q&A —
a second worked example alongside the [FINMA report](https://github.com/chengmanov/apertus-finma-eval),
on a different sector (legal, not banking) and a different document type (a
statute, not supervisory circulars).

The task: answer questions about the **Swiss Code of Obligations
(Obligationenrecht, OR / SR 220)** *and cite the exact article* the answer
rests on. Scoring is **deterministic** — answer by string/number match,
citation by exact article-number match. **No LLM judge.** Everything runs on
public data on a single 4–8B open model
([Apertus-8B-Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)).
Full write-up: **[report/REPORT.md](report/REPORT.md)**.

## Results

Four systems, same base model, quantized identically to Q4_K_M and served
identically — the only variable is the weights. 172 human-verified questions
(98 German, 74 French, 10 contract-law topics). 95% Wilson intervals.

| Metric | A · base | B · base+RAG | C · fine-tuned | D · fine-tuned+RAG |
|---|---|---|---|---|
| **Answer + citation** (production bar) | 4.1% [2–8] | 38.4% [31–46] | 2.9% [1–7] | **52.3% [45–60]** |
| Answer correct | 39.0% | 54.7% | 46.5% | **70.3%** |
| Citation correct | 7.0% | 69.8% | 5.8% | 62.2% |
| Emits the required format | 1.2% | 0.0% | 89.5% | 70.9% |

Fine-tuning + retrieval is the best system (52.3%), reproducing the FINMA
finding on an entirely different body of law.

## The failure mode this report documents

The first fine-tune was trained on **question-only** prompts (no retrieved
context). With retrieval added at inference it *collapsed* — dropping the
citation line and scoring **8.1%** on the production bar, worse than base+RAG.
The fix is best practice: **train retrieval-aware** (each item trained both
plain and with retrieved context, gold article guaranteed present). That took
arm D from 8.1% to 52.3%.

| Fine-tuned + RAG | Production bar | Format |
|---|---|---|
| Naive (question-only training) | 8.1% | 15.7% |
| Retrieval-aware training | 52.3% | 70.9% |

A model served *with* retrieval must be *trained* with retrieval — exactly the
kind of thing a Pilot's ablation catches before production.

## Why article numbers

The OR numbers every provision ("OR Art. 335"). Those article numbers are the
citation currency of Swiss contract law and give the benchmark a rare property:
citation correctness is **exactly checkable**, so the whole thing is scored by
string and number matching with no model-as-judge. An answer that is right but
cites the wrong article is scored as a **failure**, because in legal work it is
one.

## Reproduce it

```bash
uv sync
uv run python scripts/download_corpus.py de fr     # pinned OR from Fedlex (SHA-256 checked)
uv run python scripts/extract_articles.py          # article-anchored extraction
uv run python scripts/assemble_tasks.py            # build the verified eval set

# serve a Q4_K_M model with the training-consistent chat template, e.g.:
#   llama-server -m Apertus-8B-base-Q4_K_M.gguf --port 8080 \
#     --jinja --chat-template-file configs/apertus_chat_template.jinja
uv run python scripts/run_eval.py --arm B --api http://localhost:8080 --model-tag base
uv run python scripts/score_run.py runs/*.jsonl    # ablation table + Wilson CIs
```

Fine-tune (arms C/D), on an NVIDIA box:

```bash
bash train/setup_gb10.sh                            # aarch64 + Blackwell env (adapt for other GPUs)
uv run python scripts/build_sft.py                  # retrieval-aware SFT set
python train/sft_lora.py --quant none --merge       # bf16 LoRA, merge to fp16
# convert merged -> GGUF, quantize to Q4_K_M, serve, run arms C/D as above
```

## How it's built

- **Corpus** — the OR consolidation in force 2026-01-01 from the official
  [Fedlex](https://www.fedlex.admin.ch) filestore (DE + FR), SHA-256-pinned so
  an upstream change fails loudly and the corpus version is bumped deliberately.
- **Article extraction** — a streaming HTML parser over Fedlex's clean
  `<article id="art_N">` markup (`scripts/extract_articles.py`); footnote
  references are distinguished from paragraph markers, repealed/empty articles
  excluded. Fedlex's structured HTML is far cleaner than the FINMA PDFs — the
  point of the OR is to show the same method on a well-structured source too.
- **Eval set** — 172 items, authored from the statute and **independently
  verified** against the source articles (see `tasks/GUIDELINES.md`); drafts and
  verdicts retained.
- **SFT set** — authored from articles **disjoint** from the eval's gold
  citations, contamination enforced in `scripts/build_sft.py`, then rendered
  **retrieval-aware** (plain + with-context variants).
- **Harness** — BM25 retrieval over articles, an OpenAI-compatible client, a
  two-line `ANSWER:/SOURCE:` output contract, a robust parser (scores content
  wherever it appears; reports strict-format compliance separately),
  deterministic scoring, and Wilson confidence intervals.

## Layout

```
data/        manifest + checksums; raw HTML and extracted text are regenerated
tasks/       GUIDELINES.md, eval_v1.jsonl, drafts/ + verified/, SFT sets
src/legal_eval/  corpus, retrieval, client, prompts, scoring, stats
scripts/     download_corpus, extract_articles, make_slices, assemble_tasks, build_sft, run_eval, score_run
train/       sft_lora.py, setup_gb10.sh
configs/     apertus_chat_template.jinja
report/      REPORT.md + frozen run records (runs/)
```

## Legal

The Code of Obligations is an official publication of the Swiss Confederation,
publicly available at fedlex.admin.ch. This project downloads it for
research/evaluation and does not redistribute it; the pinned manifest lets
anyone re-fetch the identical corpus. Not affiliated with or endorsed by the
Swiss authorities. Nothing here is legal advice.

---

Built by [sysf.io](https://sysf.io) — sovereign, task-tuned AI for regulated
Swiss and EU organisations. Rendered version:
[sysf.io/reports/or-ablation](https://sysf.io/reports/or-ablation/).

# finma-eval — does fine-tuning a small model actually help?

An open, reproducible evaluation of a small language model on a real Swiss
regulatory task — built to replace an argument with a measurement:

> What do retrieval and fine-tuning each actually buy on a regulated task,
> measured on a small open model you can run on your own hardware?

The task: answer questions about [FINMA
circulars](https://www.finma.ch/en/documentation/circulars/) **and cite the
exact margin number (Randziffer, "Rz")** the answer rests on. Scoring is
**deterministic** — answer accuracy by string/number match, citation
faithfulness by exact Rz match. **No LLM judge anywhere.**

Everything here runs on public data on a single 4–8B open model
([Apertus-8B-Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509),
Switzerland's sovereign LLM). Full write-up: **[report/REPORT.md](report/REPORT.md)**.

## Results

Four systems, same base model, quantized identically to Q4_K_M and served
identically — the only variable is the weights. 183 human-verified questions
(101 German, 82 English, 10 circulars). Brackets are 95% Wilson intervals.

| Metric | A · base | B · base+RAG | C · fine-tuned | D · fine-tuned+RAG |
|---|---|---|---|---|
| **Answer + citation** (production bar) | 0.0% [0–2] | 22.4% [17–29] | 1.6% [1–5] | **45.4% [38–53]** |
| Answer correct | 37.2% | 63.4% | 44.8% | **80.9%** |
| Citation correct | 1.1% | 39.9% | 2.2% | **50.3%** |
| Emits the required format | 2.2% | 0.0% | 100% | **100%** |

**Fine-tuning plus retrieval doubles the production bar** (22.4% → 45.4%).
The ablation shows *why*: retrieval is what makes citation possible (you can't
cite an Rz you never retrieved), and fine-tuning is what makes the output
usable (the base model often knows the answer but emits the required,
machine-parseable format only ~2% of the time; the fine-tune, 100%). Neither
ingredient alone clears the bar — the interaction is the finding.

## Why Rz numbers matter

FINMA circulars number every normative paragraph with a margin note. Those Rz
numbers are the citation currency of Swiss financial supervision — a compliance
officer needs *"Rz 9–10, FINMA-RS 2018/03"*, not a paragraph of confident
prose. That gives the benchmark a rare property: **citation correctness is
exactly checkable**, so the whole thing is scored by string and number matching
with no model-as-judge. An answer that's right but cites the wrong Rz is scored
as a **failure**, because in regulated work it is one.

## Reproduce it

```bash
uv sync
uv run python scripts/download_corpus.py de en    # pinned FINMA circulars (SHA-256 checked)
uv run python scripts/extract_text.py             # Rz-anchored extraction
uv run python scripts/assemble_tasks.py           # build the verified eval set

# serve a Q4_K_M model with the training-consistent chat template, e.g.:
#   llama-server -m Apertus-8B-base-Q4_K_M.gguf --port 8080 \
#     --jinja --chat-template-file configs/apertus_chat_template.jinja
uv run python scripts/run_eval.py --arm B --api http://localhost:8080 --model-tag base
uv run python scripts/score_run.py runs/*.jsonl   # ablation table + Wilson CIs
```

Fine-tune (arms C/D), on an NVIDIA box — bootstrap + train:

```bash
bash train/setup_gb10.sh                          # aarch64 + Blackwell env (adapt for other GPUs)
python train/sft_lora.py --quant none --merge     # bf16 LoRA on the SFT set, merge to fp16
# convert merged -> GGUF, quantize to Q4_K_M, serve, run arms C/D as above
```

## How it's built

- **Corpus** — `data/manifest.json` pins current FINMA circulars (banking core)
  with per-language PDF URLs; downloads are SHA-256-pinned
  (`data/checksums.json`), so an upstream change fails loudly and the corpus
  version is bumped deliberately. The corpus is versioned, like everything else.
- **Rz-anchored extraction** — current FINMA PDFs render unamended margin
  numbers as tiny vector *images*, not text. `scripts/extract_text.py`
  reconstructs the numbering geometrically (margin digit-images + amended-Rz
  text anchors), with a digit-width self-check that warns on mismatch. Annex-
  and table-heavy circulars (2015/02, 2019/02, 2020/01, DE 2011/02 & 2017/01)
  are excluded from corpus v1 rather than shipped dirty — extraction limits are
  part of the methodology, not a footnote.
- **Eval set** — 183 items, authored from the circulars and **independently
  verified** against the source Rz chunks (see `tasks/GUIDELINES.md`); drafts
  and verdicts are retained under `tasks/`.
- **SFT set** — 513 train / 27 val, authored from chunks **disjoint** from the
  eval's gold citations, with contamination and eval-collision rejection
  enforced in `scripts/build_sft.py` (not just trusted from the prompt).
- **Harness** — BM25 retrieval over Rz chunks, an OpenAI-compatible client, a
  two-line `ANSWER:/SOURCE:` output contract, a robust parser that scores
  content wherever it appears (and reports strict-format compliance
  separately), deterministic scoring, and Wilson confidence intervals.
- **Fine-tune** — bf16 LoRA (rank 16, 3 epochs) on Apertus-8B-Instruct; the
  serving template (`configs/apertus_chat_template.jinja`) reproduces the
  model's actual training chat format byte-for-byte.

## Layout

```
data/        manifest + checksums; raw PDFs and extracted text are regenerated
tasks/       GUIDELINES.md, eval_v1.jsonl, drafts/ + verified/, SFT sets
src/finma_eval/  corpus, retrieval, client, prompts, scoring, stats
scripts/     download_corpus, extract_text, assemble_tasks, build_sft, run_eval, score_run
train/       sft_lora.py, setup_gb10.sh
configs/     apertus_chat_template.jinja
report/      REPORT.md + frozen run records (runs/)
```

## Legal

FINMA circulars are official publications of the Swiss Financial Market
Supervisory Authority, publicly available at finma.ch. This project downloads
them for research/evaluation and does not redistribute them; the pinned
manifest lets anyone re-fetch the identical corpus from the source. Not
affiliated with or endorsed by FINMA. Nothing here is legal advice.

---

Built by [sysf.io](https://sysf.io) — sovereign, task-tuned AI for regulated
Swiss and EU organizations. A worked version of this report:
[sysf.io/reports/finma-ablation](https://sysf.io/reports/finma-ablation/).

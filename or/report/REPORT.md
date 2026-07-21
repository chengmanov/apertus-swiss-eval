# Fine-tuning a small model for Swiss contract law — and the mistake that quietly breaks it

**A second sample evaluation report. Same method as the [FINMA report](https://github.com/chengmanov/apertus-finma-eval), a different regulatory domain — and a real failure mode we hit, diagnosed, and fixed on the way.**

## The question

Does the "fine-tune a small open model and measure it" approach generalise
beyond one dataset? We repeated it on a different sector (legal, not banking)
and a different kind of document (a statute, not supervisory circulars):
**answering questions about the Swiss Code of Obligations (Obligationenrecht,
OR / SR 220) with a citation to the exact article the answer rests on.**

Four systems, all built on the same open model
([Apertus-8B-Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)),
quantized identically to Q4_K_M and served identically:

| Arm | System |
|-----|--------|
| **A** | Base model |
| **B** | Base model + retrieval (RAG) |
| **C** | Fine-tuned model |
| **D** | Fine-tuned model + retrieval |

Both scored **deterministically — no LLM judge**: is the answer correct
(string/number match against a verified gold answer), and did it cite the
correct article (exact article-number match). The production bar is **both at
once**. The article number ("OR Art. 335") is a clean, exact citation anchor,
exactly like the FINMA margin number (Rz) — which is what makes judge-free
scoring possible.

## The headline

Scores over **172 human-verified questions** (98 German, 74 French, 10
contract-law topics). Brackets are 95% Wilson confidence intervals.

| Metric | A · base | B · base+RAG | C · fine-tuned | D · fine-tuned+RAG |
|---|---|---|---|---|
| **Answer + citation (production bar)** | 4.1% [2–8] | 38.4% [31–46] | 2.9% [1–7] | **52.3% [45–60]** |
| Answer correct | 39.0% | 54.7% | 46.5% | **70.3%** |
| Citation correct | 7.0% | 69.8% | 5.8% | 62.2% |
| Emits the required format | 1.2% | 0.0% | 89.5% | 70.9% |

**Fine-tuning plus retrieval is the best system, at 52.3%** — well above
retrieval alone (38.4%), and the method reproduces the FINMA finding on an
entirely different body of law. The decomposition is the same story: retrieval
supplies the citable text (citation jumps from ~6% to ~62–70% only with RAG),
fine-tuning supplies the disciplined, machine-parseable output, and only the
two together clear the bar.

## The mistake worth publishing

We didn't get 52.3% on the first try. Our **first** fine-tune was trained the
obvious way — on questions and their answer+citation, *without* retrieved
context. On the no-retrieval arm it looked great (89% well-formatted). But with
retrieval added, it **collapsed**:

| Fine-tuned + RAG (arm D) | Production bar | Emits format |
|---|---|---|
| Naive fine-tune (question-only training) | **8.1%** | 15.7% |
| Retrieval-aware fine-tune | **52.3%** | 70.9% |

The naive model, shown retrieved passages it had never seen during training,
reverted to terse answering and **dropped the citation line** — worse on the
production bar than the base model with retrieval (38.4%). Same question, same
retrieved context:

> **Q:** *Ist eine im Voraus getroffene Abrede gültig, mit der die Haftung für grobe Fahrlässigkeit wegbedungen wird?* (gold: **nein**, Art. 100)
> **Naive fine-tune + RAG:** `ANSWER: nein` — right answer, **no citation** → fails
> **Retrieval-aware fine-tune + RAG:** `ANSWER: nein` / `SOURCE: OR Art. 100` → passes

The cause is a **training/inference distribution mismatch**: a model that will
be served *with* retrieval must be *trained* with retrieval. We rebuilt the
fine-tuning set so every example appears both plain and with retrieved context
(the gold article guaranteed present), retrained, and the production bar went
from 8.1% to 52.3%. This is the kind of thing a real Pilot's ablation catches
before it reaches production — and the reason we measure instead of assume.
(The FINMA fine-tune happened to be robust to this; the OR one was not. Both
are now trained retrieval-aware.)

## The honest limits

- **List-answer questions score 0% across every arm**, including the winner.
  Extracting a complete multi-item set (e.g. "name the elements of X") and
  citing it is the hardest sub-task here and the clear next target — the same
  weak spot the FINMA report flagged.
- **French trails German** on the production bar (D: 42% FR vs 60% DE),
  tracking the training mix (271 DE vs 123 FR fine-tuning items). More balanced
  data closes this; the number is honest about where the current model is weak.
- **Format compliance is 71%, not 100%.** The retrieval-aware fine-tune still
  drops the citation on some items — better than the naive 16%, not perfect.
  A larger or more balanced training set moves it further.
- **45–52% is a floor, not a ceiling** — one 4–8B model at 4-bit, a short LoRA
  run, deliberately plain BM25 retrieval. The value is the method and the
  deltas between arms, not a leaderboard number.

## Why the OR is a good second test

The FINMA report showed the method on supervisory circulars, where the citation
anchor is a margin number (Rz). The Code of Obligations is a different shape of
regulatory text — a consolidated statute with ~1,150 numbered articles — and
its article numbers ("OR Art. 335") give the same judge-free, exactly-checkable
citation target. The corpus is public (the official [Fedlex](https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de)
consolidation), version-pinned by checksum, and the task set was built from the
statute and independently verified. Everything here is reproducible from the
[repository](https://github.com/chengmanov/apertus-legal-eval).

## Reproduce it

```bash
uv sync
uv run python scripts/download_corpus.py de fr     # pinned OR from Fedlex
uv run python scripts/extract_articles.py          # article-anchored extraction
uv run python scripts/assemble_tasks.py            # verified eval set
# serve a Q4_K_M model with configs/apertus_chat_template.jinja, then:
uv run python scripts/run_eval.py --arm D --api http://localhost:8080
uv run python scripts/score_run.py runs/*.jsonl    # ablation table + Wilson CIs
```

Corpus: the Code of Obligations, consolidation in force 2026-01-01, DE + FR,
SHA-256-pinned. Eval set: 172 items, each verified against its source article.
Fine-tune: retrieval-aware bf16 LoRA on Apertus-8B-Instruct (each item trained
plain and with retrieved context). Training and evaluation ran on-premise — no
data left the machines, which is the whole point.

---

*This is a sample of the deliverable sysf.io produces in a Pilot: your task,
your documents, this ablation — including the failure modes it catches — before
you commit to production. [sysf.io](https://sysf.io)*


---

## Second iteration (v2)

v2 changed **only the hyperparameters** (LoRA r=32 / 4 epochs vs r=16 / 3 epochs) on the *same* v1 SFT — a deliberate control: more capacity, identical data. FINMA and OR already had genuine human-verified Q&A in v1, so the data lever did not apply.

**Arm D (fine-tuned + retrieval), v1 → v2** — same 172-item eval set, same deterministic scorer:

| Metric | v1 | v2 | Δ |
|--------|---:|---:|---:|
| Answer + citation (production bar) | 52.3% | **51.7%** | -0.6 |
| Answer correct | 70.3% | **71.5%** | +1.2 |
| Citation correct | 62.2% | **61.0%** | -1.2 |
| Emits required format | 70.9% | **73.8%** | +2.9 |

Essentially **flat** (52.3% → 51.7%, n=172): more capacity on the same data bought nothing. The lever is the data, not the training budget.

*Full method for the second iteration: [`../METHODOLOGY.md`](../METHODOLOGY.md). v2 run records are in `runs/` (`*-v2`).*


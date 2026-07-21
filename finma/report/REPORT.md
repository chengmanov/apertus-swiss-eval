# Does fine-tuning a small model actually help? A measured answer on FINMA circular Q&A

**A sample evaluation report — the kind of ablation sysf.io runs at the start of every Pilot, here on public data so anyone can reproduce it.**

## The question

Regulated teams are told two contradictory things about small open models: that they're "good enough now," and that they "can't be trusted for compliance work." Neither is a number. This report replaces the argument with a measurement on a concrete, verifiable task: **answering questions about Swiss FINMA circulars with a citation to the exact margin number (Randziffer, "Rz") the answer rests on.**

We measure four systems, all built on the same 4–8B open model ([Apertus-8B-Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), Switzerland's sovereign LLM), quantized identically to Q4_K_M and served identically:

| Arm | System |
|-----|--------|
| **A** | Base model |
| **B** | Base model + retrieval (RAG) |
| **C** | Fine-tuned model |
| **D** | Fine-tuned model + retrieval |

Two things are scored, both **deterministically — no LLM judge**:

- **Answer** — is the answer factually correct? (exact/normalized match against a gold answer)
- **Citation** — did it cite the correct Rz in the correct circular? (exact Rz match)

The bar that matters for production is **both at once**: a correct answer with a fabricated or missing citation is a failure, because in regulated work it is one.

## The headline

Scores over **183 human-verified questions** (101 German, 82 English, 10 circulars). Brackets are 95% Wilson confidence intervals.

| Metric | A · base | B · base+RAG | C · fine-tuned | D · fine-tuned+RAG |
|---|---|---|---|---|
| **Answer + citation (production bar)** | 0.0% [0–2] | 22.4% [17–29] | 1.6% [1–5] | **45.4% [38–53]** |
| Answer correct | 37.2% [30–44] | 63.4% [56–70] | 44.8% [38–52] | **80.9% [75–86]** |
| Citation correct | 1.1% [0–4] | 39.9% [33–47] | 2.2% [1–5] | **50.3% [43–57]** |
| Emits the required format | 2.2% | 0.0% | 100% | **100%** |

**Fine-tuning plus retrieval doubles the production bar** — from 22.4% (base+RAG) to 45.4%. On raw answer accuracy it goes from 63% to 81%. And it takes format compliance from ~nothing to perfect.

## What each layer actually buys

The ablation exists to stop anyone — including us — from hand-waving. Read across the four columns and the contribution of each layer is unambiguous:

**Retrieval is what makes citation possible at all.** Compare citation-correct: base 1.1% → base+RAG 39.9%; fine-tuned 2.2% → fine-tuned+RAG 50.3%. A model cannot cite a margin number it has never seen. Without retrieval, both models are near zero on the production bar (A: 0.0%, C: 1.6%) no matter how well they're trained. *If a vendor promises grounded citations without retrieval, that number above is what they're really selling.*

**Fine-tuning is what makes the output usable.** The base model, given the retrieved text, often knows the answer — 63% answer-correct in arm B — but it will not reliably produce it in the disciplined, machine-parseable form a downstream system needs. Its strict-format compliance is **2.2%, and drops to 0.0% once retrieval fills the context** (the extra text distracts it further). The fine-tuned model emits the exact `ANSWER: … / SOURCE: FINMA-RS … Rz N` contract **100%** of the time. That single change is most of the gap between B and D.

**The two compound.** Neither alone clears the production bar; together they reach 45.4%. That interaction — not either ingredient alone — is the finding.

### It looks like this

Same question, same retrieved context, base+RAG vs. fine-tuned+RAG:

> **Q:** *For banks, securities firms and financial groups, what is the minimum required equity capital at which the fully qualitative and quantitative requirements apply?* (gold: **CHF 10 billion**, Rz 6, FINMA-RS 2010/01)
> **Base + RAG:** `CHF 10 billion` — right answer, no citation → **fails the production bar**
> **Fine-tuned + RAG:** `ANSWER: CHF 10 billion` / `SOURCE: FINMA-RS 2010/01 Rz 6` → **passes**

> **Q:** *A firm calculates the bonuses of its risk-control staff directly on the basis of trading results — is this permitted?* (gold: **no**, Rz 59)
> **Base + RAG:** `no; Rz 59 (IV. Principles) states that the calculation of variable remuneration of these persons must not be d…` — correct and even cites, but the free-form prose is not machine-parseable and mixes answer with source
> **Fine-tuned + RAG:** `ANSWER: no` / `SOURCE: FINMA-RS 2010/01 Rz 59`

The base model is not stupid. It is *unreliable in shape* — and reliability of shape is exactly what separates a demo from a system.

## The honest limits

- **45% is not 95%.** This is a hard task (short-form answer *and* exact citation, single 4–8B model, Q4_K_M, one modest LoRA run on 513 examples, no retrieval tuning). It is a floor, not a ceiling: retrieval quality, a larger model for the escalation tier, and more training data each move it. The point of the report is the *method and the deltas*, not a leaderboard number.
- **Structured answer types are harder.** By answer type, fine-tuned+RAG scores 47–50% on numbers/durations/yes-no but 27% on multi-item list questions — extraction of complete sets is the weakest area and the obvious next target.
- **These numbers supersede an earlier draft run** that used a simplified serving template and a third-party quantization. Both were replaced so that all four arms use the model's *actual* training chat format and byte-identical quantization — the only variable left is the weights. Rigor cuts against convenient numbers sometimes; that's the job.
- **Retrieval here is deliberately plain** (BM25 over Rz-chunked text, no re-ranking, no embeddings). A production Pilot tunes this; we kept it transparent so the ablation isolates fine-tuning rather than retrieval cleverness.

## Why this is the right task to measure on

FINMA circulars number every normative paragraph with a margin number, and those Rz numbers are the citation currency of Swiss financial supervision — *"Rz 9–10, FINMA-RS 2018/03"*, not a paragraph of confident prose. That gives the evaluation a rare property: **citation correctness is exactly checkable**, so the whole benchmark is scored by string and number matching with no model-as-judge in the loop. The documents are public, the corpus is version-pinned, and the task set was built from the documents and independently verified. Everything here is reproducible from the [repository](https://github.com/chengmanov/apertus-finma-eval).

## Reproduce it

```bash
uv sync
uv run python scripts/download_corpus.py de en     # pinned FINMA circulars
uv run python scripts/extract_text.py              # Rz-anchored extraction
# serve each Q4_K_M model with configs/apertus_chat_template.jinja, then:
uv run python scripts/run_eval.py --arm D --api http://localhost:8080
uv run python scripts/score_run.py runs/*.jsonl    # ablation table + CIs
```

Corpus: 17 current FINMA circulars, SHA-256-pinned. Eval set: 183 items, every one verified against its source paragraph. Fine-tune: bf16 LoRA (rank 16, 3 epochs, 513 contamination-controlled examples) on Apertus-8B-Instruct. Training and evaluation ran on-premise — no data left the machines, which is the whole point.

---

*This is a sample of the deliverable sysf.io produces in a Pilot: your task, your documents, this ablation, before you commit to production. [sysf.io](https://sysf.io)*


---

## Second iteration (v2)

v2 changed **only the hyperparameters** (LoRA r=32 / 4 epochs vs r=16 / 3 epochs) on the *same* v1 SFT — a deliberate control: more capacity, identical data. FINMA and OR already had genuine human-verified Q&A in v1, so the data lever did not apply.

**Arm D (fine-tuned + retrieval), v1 → v2** — same 183-item eval set, same deterministic scorer:

| Metric | v1 | v2 | Δ |
|--------|---:|---:|---:|
| Answer + citation (production bar) | 45.4% | **33.9%** | -11.5 |
| Answer correct | 80.9% | **78.1%** | -2.8 |
| Citation correct | 50.3% | **41.0%** | -9.3 |
| Emits required format | 100.0% | **100.0%** | 0 |

The production bar **fell** 45.4% → 33.9% (n=183). More capacity on the same data **over-fit** — direct evidence that the v2 gains elsewhere come from the *data*, not the training budget.

*Full method for the second iteration: [`../METHODOLOGY.md`](../METHODOLOGY.md). v2 run records are in `runs/` (`*-v2`).*


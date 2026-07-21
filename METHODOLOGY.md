# Methodology

How every number in this repository is produced — and, for the second iteration
(v2), exactly what changed and why. The goal is that a skeptical reader can
reproduce each figure and see that nothing is hand-tuned to flatter the result.

## The task

For each domain, a fixed set of short factual questions about a Swiss legal text.
A system must return **two lines**:

```
ANSWER: <short answer>
SOURCE: <LAW> Art. <number>        (FINMA: Rz <n>, circular <id>)
```

Two things are scored, both **deterministically — no LLM judge**:

- **Answer** — normalized string / number match against a verified gold answer.
- **Citation** — exact match of the legal unit (article number, or FINMA margin
  number Rz) against the gold unit.

The **production bar** is *answer AND citation correct*. A right answer with the
wrong citation is a failure, because in regulated work it is one. Strict two-line
format compliance is measured and reported **separately** — it is never used to
zero out a domain-correct answer, so the harness cannot inflate itself.

All figures are recomputed from **frozen per-item run records** (`*/report/runs/`)
with 95 % Wilson confidence intervals.

## The four arms

Same open model (Apertus-8B), quantized identically to 4-bit (Q4_K_M, no imatrix)
and served identically. The only variables are retrieval and the LoRA fine-tune.

| Arm | Retrieval | Model |
|-----|-----------|-------|
| A | — | base |
| B | BM25 (k) over the corpus | base |
| C | — | LoRA fine-tuned |
| D | BM25 (k) over the corpus | LoRA fine-tuned |

## Corpus

Pinned and SHA-256-checksummed from the official source — the [Fedlex](https://www.fedlex.admin.ch)
filestore for the five statutes, FINMA for the circulars (`*/data/manifest.json`,
`*/data/checksums.json`). A silently changed upstream fails loudly. Text is
chunked by legal unit (article, or FINMA margin number). Extraction code lives in
each domain's `scripts/`.

## Eval set (held FIXED across v1 and v2)

Short-answer questions with a machine-checkable gold answer and a gold citation,
in German and (for the statutes) French. **The eval set does not change between
iterations** — so any v1 → v2 movement is attributable to the training changes,
not to an easier benchmark. Provenance differs by domain and is stated honestly:
FINMA and OR were built with an independent human-verification pass; the four
newer statutes (DSG, HMG, VVG, PrSG) were authored from the source article text
and verified against it.

## Scoring

`src/<law>_eval/scoring.py`, identical across arms and iterations: answer by
normalized string/number match (with per-item accepted variants); citation by
exact unit match against `gold_article` (+ any `also_acceptable_article`). The
scorer is **not** changed between v1 and v2.

---

## v1 (first iteration)

- **Retrieval:** BM25, k = 5.
- **Fine-tune:** LoRA r=16, α=32, 3 epochs, on Apertus-8B-Instruct, RAG-aware
  (each SFT item trained both plain and with retrieved context, gold unit
  guaranteed present) — this fixes the train/inference mismatch that otherwise
  makes the fine-tuned model drop the SOURCE line under retrieval.
- **SFT data:**
  - FINMA, OR — genuine short-answer Q&A (human-verified), a few hundred items.
  - DSG, HMG, VVG, PrSG — **topic/section-identification** items (answer = the
    article's marginal note or section heading) plus a handful of hand-written
    factual items, contamination-controlled against the eval articles.

**What v1 established:** retrieval is necessary (a small model cannot cite a unit
it has never seen), and fine-tuning *on top of* retrieval (arm D) is the winning
system. But for the four newer domains the topic-ID SFT taught *format* and
*which article covers which topic* — not how to *answer* — so arm C (fine-tuned,
no retrieval) often **regressed** answer accuracy. That is the gap v2 targets.

## v2 (second iteration) — what changed and why

The eval set, the scorer, the base model and its 4-bit quantization are all held
fixed. Three levers change, each documented so its effect is legible:

### 1. Task-matched SFT data (primary lever)

For the four newer domains we replace the topic-ID-heavy SFT with **genuine short
factual Q&A that matches the eval distribution**, generated as follows:

- **Generation:** Claude authored 2–6 short factual questions per *non-eval*
  article (contamination-controlled), each with a short machine-checkable answer
  and the gold article it comes from, in the article's language.
- **Deterministic grounding filter** (`filter_grounded.py`, no LLM judge): an item
  is kept only if its answer is provably supported by its gold article's text —
  for `number`/`duration` every digit (or spelled-out numeral) in the answer must
  appear in the article; for `short` a content word of the answer must appear in
  the article; `yesno` must be ja/nein/oui/non. Ungrounded items, eval-article
  contamination, and duplicate questions are dropped. Kept/dropped counts are
  recorded per domain.
- The surviving factual items are **blended with the v1 topic-ID items** (which
  still carry the article↔topic citation signal) and rebuilt RAG-aware.
- FINMA and OR keep their v1 human-verified SFT (adding synthetic data there would
  lower, not raise, their quality bar); they receive only levers 2 and 3.

### 2. Retrieval (tested, then reverted)

We tried raising BM25 depth **k = 5 → 8**. On the first domain evaluated (PrSG) it
**did not help and slightly hurt** — arm B citation fell (more retrieved passages
diluted the gold unit), while the production bar was unchanged. So v2 **holds
retrieval at k = 5, identical to v1**, and arms A and B are **reused unchanged from
v1**. This is deliberate: it means every v1 → v2 movement in arms C and D is
attributable to the *training* change alone, with no retrieval confound. (Reported
as a negative result, per the guardrails below.)

### 3. Fine-tune hyperparameters

LoRA **r=16 → 32, α=32 → 64, 3 → 4 epochs**, same RAG-aware construction, same
`loss_type="nll"`, single-GPU (`device_map={"":0}`). Trained on the NVIDIA GB10
(DGX Spark) cluster, two domains at a time; quantized to Q4_K_M with the same
no-imatrix toolchain as the base arm.

### Honesty guardrails

- The eval set and scorer are unchanged, so v1→v2 deltas are real.
- Synthetic SFT is a *training aid* only; it never becomes eval gold, and it is
  contamination-filtered against the eval articles.
- If a domain does not improve, the v2 table shows that. A null result is a result.

Per-domain v1→v2 tables (with the exact SFT counts and CIs) are in each
`*/report/REPORT.md`; the headline comparison is in the top-level `README.md`.

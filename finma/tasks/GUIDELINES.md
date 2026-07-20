# Task authoring guidelines (eval set v1)

Every item is a short-form factual question about the normative content of a
FINMA circular, answerable from the pinned corpus, scored deterministically.
These guidelines are part of the published methodology.

## Item schema (JSON)

```json
{
  "id": "assigned at assembly",
  "lang": "de | en",
  "circular": "2018/03",
  "slug": "2018-03",
  "question": "Darf eine Bank ihre Compliance-Funktion vollständig auslagern?",
  "answer_type": "yesno | number | duration | short | list",
  "gold": "nein",
  "accept": ["no", "nicht zulässig"],
  "gold_rz": ["9"],
  "also_acceptable_rz": ["10"],
  "section": "IV. Zulässigkeit",
  "topic": "outsourcing",
  "difficulty": "basic | applied"
}
```

## Rules

1. **Answerable from one Rz chunk.** The question's answer must be fully
   contained in the chunk(s) listed in `gold_rz`. Multi-Rz questions are
   allowed only when the Rz are adjacent and `gold_rz` lists all of them.
2. **Machine-checkable answers only.**
   - `yesno`: gold is "yes"/"no" ("ja"/"nein" for DE) — the *condition* goes
     in the question, not the answer. Bad: "yes, if the board approves".
     Good question: "Does outsourcing the compliance function require …?"
   - `number` / `duration`: a single number or duration ("30", "5 Jahre",
     "10 Geschäftstage"). Put units in `gold`; list unit variants in `accept`.
   - `short`: a noun phrase of at most 6 words with essentially one correct
     formulation; list spelling/wording variants in `accept`.
   - `list`: 2–5 short items; scored as an unordered set with per-item
     matching; each list element gets its own accept-variants
     (`gold` is an array, `accept` an array of arrays).
3. **Standalone and natural.** Phrase questions the way a compliance officer
   or auditor would ask them, naming the subject matter (not "per Rz 12,
   what…" and not "according to this circular…"). The question must be
   unambiguous *across the whole corpus* — if two circulars could plausibly
   answer it, name the institution type or context that disambiguates.
4. **No string-overlap giveaways.** Don't copy a full distinctive sentence
   from the source into the question. Paraphrase. A BM25 query built from the
   question SHOULD still retrieve the right chunk — that's realistic — but
   the answer token itself must not appear in the question.
5. **Language-pure.** DE questions get DE gold answers, EN questions EN gold.
   Author DE items from `_de` extractions and EN items from `_en` extractions.
6. **No empty-chunk references.** If the source chunk text is empty (table or
   figure in the original), skip it.
7. **Spread coverage.** Within a circular, spread items across sections; avoid
   clustering five items on one Rz. Tag `difficulty: applied` when the
   question wraps the rule in a realistic scenario ("A bank wants to …, may
   it …?") rather than restating it.
8. **Citation is part of the task.** At evaluation time the model is asked to
   answer AND cite the Rz. An item's `gold_rz` (plus `also_acceptable_rz` for
   directly adjacent, equally-supporting paragraphs) defines citation
   correctness. Be strict: a chunk that merely mentions the topic does not
   support the answer.

## Verification (second pass, independent)

Every drafted item is verified by a reviewer who did not author it, against
the source chunk text:

- gold answer correct and complete per the cited chunk;
- `gold_rz` supports the answer (not just the topic);
- question standalone, natural, unambiguous, language-pure;
- answer machine-checkable under the type rules above;
- verdict: `keep`, `fix` (with corrected fields), or `drop` (with reason).

Only `keep`/`fix` items enter the eval set. Drop reasons are retained for the
methodology appendix.

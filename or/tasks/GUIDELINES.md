# Task authoring guidelines (OR eval set v1)

Every item is a short-form factual question about the Swiss Code of Obligations
(OR / SR 220), answerable from the pinned corpus, scored deterministically.
These guidelines are part of the published methodology.

## Item schema (JSON)

```json
{
  "id": "assigned at assembly",
  "lang": "de | fr",
  "law": "OR",
  "question": "Kann ein unbefristetes Arbeitsverhältnis von jeder Partei gekündigt werden?",
  "answer_type": "yesno | number | duration | short | list",
  "gold": "ja",
  "accept": ["yes"],
  "gold_article": ["335"],
  "also_acceptable_article": [],
  "section": "Arbeitsvertrag / Beendigung",
  "topic": "employment",
  "difficulty": "basic | applied"
}
```

## Rules

1. **Answerable from one article.** The answer must be fully contained in the
   article(s) listed in `gold_article`. Multi-article questions only when the
   articles are directly adjacent/complementary and all are listed.
2. **Cite the article number, not the paragraph.** `gold_article` holds the
   article number(s), e.g. `["335"]`, `["266a"]`, `["335a"]` (letter suffixes
   are part of the number). Do not put paragraph numbers ("Abs. 2") in the
   citation — scoring is by article.
3. **Machine-checkable answers only.**
   - `yesno`: gold is "ja"/"nein" (DE) or "oui"/"non" (FR); the *condition*
     goes in the question, not the answer.
   - `number` / `duration`: a single number or period ("30", "10 Jahre",
     "3 Monate"). Units in `gold`; unit variants in `accept`.
   - `short`: a noun phrase (≤6 words) with essentially one correct wording;
     list spelling/wording variants in `accept`.
   - `list`: 2–5 short items, scored as an unordered set (`gold` an array,
     `accept` an array of arrays).
4. **Standalone and natural.** Phrase questions the way a lawyer, HR manager or
   contract reviewer would — naming the subject matter, not "per Art. 335…".
   The question must be unambiguous across the whole OR: if several articles
   could plausibly answer it, add the context (contract type, party) that
   disambiguates.
5. **No string-overlap giveaways.** Don't copy a distinctive full sentence from
   the article into the question; paraphrase. The answer token must not appear
   verbatim in the question.
6. **Language-pure.** DE questions get DE gold answers (author from `or_de`);
   FR questions get FR gold answers (author from `or_fr`).
7. **Skip repealed/empty articles.** If an article's text is empty (repealed),
   do not use it.
8. **Spread coverage.** Spread items across the articles in your assigned
   topic; avoid clustering on one article. Tag `difficulty: applied` when the
   question wraps the rule in a realistic scenario ("An employer wants to …,
   may they …?").
9. **Citation is part of the task.** At evaluation the model must answer AND
   cite the article. `gold_article` (plus `also_acceptable_article` for a
   directly adjacent, equally-supporting article) defines citation
   correctness. Be strict: an article that merely mentions the topic does not
   support the answer.

## Verification (independent second pass)

Every drafted item is checked by a reviewer who did not author it, against the
source article text: gold answer correct and complete; `gold_article` actually
supports the answer; question standalone/natural/unambiguous/language-pure;
answer machine-checkable. Verdict: `keep`, `fix` (corrected fields), or `drop`
(with reason). Only keep/fix items enter the eval set.

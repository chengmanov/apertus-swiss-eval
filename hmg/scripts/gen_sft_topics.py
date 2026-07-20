"""Generate grounded topic-identification SFT drafts from the HMG corpus.

For every article NOT used as gold in the eval set (contamination control) we
emit short-answer items whose gold is the article's official marginal note
(heading text, verbatim from the pinned Fedlex HTML). These teach the two-line
ANSWER/SOURCE format and the article<->topic association citation depends on --
without leaking any eval article. Factual answer-style supervision is added by
hand in tasks/sft_drafts/factual_<lang>.json.

Output: tasks/sft_drafts/topics_<lang>.json.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DRAFTS = ROOT / "tasks" / "sft_drafts"
EXCL = set(json.loads((ROOT / "tasks" / "sft_exclusions.json").read_text(encoding="utf-8"))["articles"])

HEADING = re.compile(r'<b>Art\.(?:\s|&nbsp;)*([0-9]+[a-z]?)</b>(?:\s|&nbsp;)*([^<]*?)\s*</a>', re.S)

Q = {
    "de": [
        "Welchen Regelungsgegenstand behandelt Artikel {n} des Heilmittelgesetzes?",
        "Unter welcher Sachüberschrift steht Artikel {n} HMG?",
    ],
    "fr": [
        "Quel objet est réglé par l'article {n} de la loi sur les produits thérapeutiques?",
        "Sous quel intitulé figure l'article {n} LPTh?",
    ],
}


def marginal_notes(html: str) -> dict[str, str]:
    notes: dict[str, str] = {}
    for art, note in HEADING.findall(html):
        note = re.sub(r"\s+", " ", note).strip()
        if art not in notes and note:
            notes[art] = note
    return notes


def main() -> int:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    for lang in ("de", "fr"):
        html = (RAW / f"hmg_{lang}.html").read_text(encoding="utf-8")
        notes = marginal_notes(html)
        items = []
        for art, note in notes.items():
            if art in EXCL or len(note) < 3 or len(note) > 90:
                continue
            for phrasing in Q[lang]:
                items.append({
                    "lang": lang, "law": "HMG",
                    "question": phrasing.format(n=art),
                    "answer_type": "short",
                    "gold": note, "accept": [note, note.lower()],
                    "gold_article": [art], "also_acceptable_article": [],
                    "topic": "topic-id", "difficulty": "basic",
                })
        (DRAFTS / f"topics_{lang}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"topics_{lang}: {len(items)} items (from {len(notes)} articles, {len(EXCL)} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

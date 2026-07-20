"""Generate grounded section-identification SFT drafts from the VVG corpus.

The VVG (unlike the DSG/HMG) carries almost no per-article marginal notes in
its Fedlex HTML, so instead of the heading we use the article's official
*section* path (Abschnitt/Kapitel), taken verbatim from the extraction, as the
grounded short answer. This still teaches the two-line ANSWER/SOURCE format and
the article<->topic association citation depends on, without leaking any eval
article.

Output: tasks/sft_drafts/topics_<lang>.json.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "text"
DRAFTS = ROOT / "tasks" / "sft_drafts"
EXCL = set(json.loads((ROOT / "tasks" / "sft_exclusions.json").read_text(encoding="utf-8"))["articles"])

Q = {
    "de": [
        "In welchem thematischen Abschnitt des Versicherungsvertragsgesetzes ist Artikel {n} eingeordnet?",
        "Zu welchem Sachbereich des VVG gehört Artikel {n}?",
    ],
    "fr": [
        "Dans quelle section thématique de la loi sur le contrat d'assurance l'article {n} est-il classé?",
        "À quel domaine de la LCA appartient l'article {n}?",
    ],
}


def leaf_section(section: str) -> str:
    """Last element of the 'A / B / C' heading path, footnote digits stripped."""
    leaf = section.split(" / ")[-1] if section else ""
    return re.sub(r"\d+$", "", leaf).strip()


def main() -> int:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    for lang in ("de", "fr"):
        data = json.loads((TEXT / f"vvg_{lang}.json").read_text(encoding="utf-8"))
        items = []
        for c in data["chunks"]:
            art = c["article"]
            if art in EXCL or not c["text"]:
                continue
            note = leaf_section(c.get("section", ""))
            if len(note) < 4 or len(note) > 90:
                continue
            for phrasing in Q[lang]:
                items.append({
                    "lang": lang, "law": "VVG",
                    "question": phrasing.format(n=art),
                    "answer_type": "short",
                    "gold": note, "accept": [note, note.lower()],
                    "gold_article": [art], "also_acceptable_article": [],
                    "topic": "section-id", "difficulty": "basic",
                })
        (DRAFTS / f"topics_{lang}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"topics_{lang}: {len(items)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

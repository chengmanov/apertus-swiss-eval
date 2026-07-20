"""Write per-topic article slices so authoring agents read small, focused files.

Topics map to article-number ranges over the OR. Letter-suffixed articles
(266a, 335b, ...) sort within their base number. Output:
data/text/slices/<topic>_<lang>.json  ({"topic", "lang", "chunks": [...]}).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "text"
OUT = TEXT / "slices"

# topic -> (low, high) inclusive article-number range (base integer)
TOPICS = {
    "formation":   (1, 40),     # conclusion, form, defects of consent, agency
    "torts":       (41, 61),    # obligations from unlawful acts
    "enrichment":  (62, 67),    # unjust enrichment
    "performance": (68, 96),    # performance of obligations
    "breach":      (97, 113),   # non-performance and default
    "extinction":  (114, 142),  # extinction, set-off, prescription
    "sale":        (184, 215),  # sale of goods and warranty
    "lease":       (253, 274),  # lease / tenancy
    "employment":  (319, 350),  # individual employment contract
    "mandate":     (394, 415),  # mandate / agency contract
}


def base_num(article: str) -> int:
    m = re.match(r"(\d+)", article)
    return int(m.group(1)) if m else 0


def main(langs: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    langs = langs or ["de", "fr"]
    for lang in langs:
        data = json.loads((TEXT / f"prsg_{lang}.json").read_text(encoding="utf-8"))
        chunks = [c for c in data["chunks"] if c["text"]]
        for topic, (lo, hi) in TOPICS.items():
            sel = [c for c in chunks if lo <= base_num(c["article"]) <= hi]
            (OUT / f"{topic}_{lang}.json").write_text(
                json.dumps({"topic": topic, "lang": lang, "chunks": sel}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  {topic}_{lang}: {len(sel)} articles (Art. {lo}-{hi})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

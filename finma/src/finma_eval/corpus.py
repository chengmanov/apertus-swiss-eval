"""Load the Rz-chunked corpus produced by scripts/extract_text.py."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEXT = ROOT / "data" / "text"

# Circulars whose extraction passed QA for corpus v1 (see README).
CORPUS_V1 = {
    "de": ["2010-01", "2011-01", "2013-08", "2016-07", "2018-03", "2019-01",
           "2023-01", "2025-01", "2025-02", "2025-04"],
    "en": ["2010-01", "2011-02", "2013-08", "2016-07", "2017-01", "2018-02",
           "2018-03", "2023-01", "2025-04"],
}


@dataclass(frozen=True)
class Chunk:
    slug: str          # "2018-03"
    circular: str      # "2018/03"
    lang: str
    rz: str
    section: str
    text: str

    @property
    def ref(self) -> str:
        return f"FINMA-RS {self.circular} Rz {self.rz}"


def load_chunks(lang: str, slugs: list[str] | None = None) -> list[Chunk]:
    slugs = slugs or CORPUS_V1[lang]
    chunks: list[Chunk] = []
    for slug in slugs:
        path = TEXT / f"{slug}_{lang}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing extraction {path}; run scripts/extract_text.py")
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data["chunks"]:
            if not c["text"]:
                continue  # tables/figures in the original
            chunks.append(Chunk(slug=slug, circular=slug.replace("-", "/"), lang=lang,
                                rz=c["rz"], section=c["section"], text=c["text"]))
    return chunks

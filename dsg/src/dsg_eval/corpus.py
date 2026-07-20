"""Load the article-chunked OR corpus produced by scripts/extract_articles.py."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEXT = ROOT / "data" / "text"


@dataclass(frozen=True)
class Chunk:
    lang: str
    law: str          # "DSG"
    article: str      # "335", "335a"
    section: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.law} Art. {self.article}"


def load_chunks(lang: str) -> list[Chunk]:
    path = TEXT / f"dsg_{lang}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing extraction {path}; run scripts/extract_articles.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for c in data["chunks"]:
        if not c["text"]:
            continue
        chunks.append(Chunk(lang=lang, law=data["law"], article=c["article"],
                            section=c.get("section", ""), text=c["text"]))
    return chunks

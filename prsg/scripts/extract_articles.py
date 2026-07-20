"""Extract article-chunked text from the Fedlex OR HTML.

Fedlex serves clean, well-structured HTML: each article is an
`<article id="art_N">` element with a `<b>Art. N</b>` heading and
`<p class="absatz">` paragraphs (paragraph numbers marked by a leading
`<sup>`). Section context (Part / Title / Chapter / Section headings) precedes
each article as `<div class="heading">` / `<hN>` elements.

For each article we emit one chunk: the article number (e.g. "335", "335a"),
the surrounding heading path, and the paragraph text with footnotes removed.
This is deliberately simpler than the FINMA Rz reconstruction — the point of
using the OR is to show the same method on a cleanly structured source.

Output: data/text/or_<lang>.json
  {"lang", "law": "PrSG", "chunks": [{"article", "section", "text"}...]}

Usage: uv run python scripts/extract_articles.py [de fr it]
"""

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "text"

ART_ID = re.compile(r"art_(\d+[a-z]?)")


class ORParser(HTMLParser):
    """Streaming parser that tracks the heading path and captures per-article text."""

    def __init__(self):
        super().__init__()
        self.articles: list[dict] = []
        self.headings: dict[int, str] = {}   # hN level -> current heading text
        self.cur = None                       # current article dict being filled
        self.depth = 0                        # <article> nesting guard
        self.in_footnote = 0
        self.in_heading_tag = None            # hN level currently open (for headings text)
        self._buf: list[str] = []             # text buffer for current context
        self._para_sup = False                # inside a <sup>
        self._in_anchor = 0                   # inside an <a> (footnote refs live here)

    # --- helpers
    def _flush_heading(self, level: int):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if not text:
            return
        # drop deeper headings when a higher-level one changes
        self.headings[level] = text
        for l in list(self.headings):
            if l > level:
                self.headings.pop(l, None)

    def _section_path(self) -> str:
        return " / ".join(self.headings[l] for l in sorted(self.headings))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "article":
            m = ART_ID.match(a.get("id", ""))
            if m:
                # close any open article first
                if self.cur:
                    self._close_article()
                self.cur = {"article": m.group(1), "section": self._section_path(), "text_parts": []}
                self.depth = 1
                return
            if self.cur:
                self.depth += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.cur is None:
            self.in_heading_tag = int(tag[1])
            self._buf = []
        elif tag == "div" and "footnotes" in a.get("class", ""):
            self.in_footnote += 1
        elif tag == "sup":
            self._para_sup = True
        elif tag == "a":
            self._in_anchor += 1

    def handle_endtag(self, tag):
        if tag == "article" and self.cur is not None:
            self.depth -= 1
            if self.depth <= 0:
                self._close_article()
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.in_heading_tag:
            self._flush_heading(self.in_heading_tag)
            self.in_heading_tag = None
        elif tag == "div" and self.in_footnote:
            self.in_footnote -= 1
        elif tag == "sup":
            self._para_sup = False
        elif tag == "a" and self._in_anchor:
            self._in_anchor -= 1

    def handle_data(self, data):
        if self.in_footnote:
            return
        if self.in_heading_tag is not None and self.cur is None:
            # heading text, but skip the "Art. N" label itself
            self._buf.append(data)
            return
        if self.cur is not None:
            if self._para_sup:
                # A bare <sup>N</sup> marks a paragraph number (Abs. N). A
                # <sup><a>N</a></sup> is a footnote reference — skip it.
                if self._in_anchor:
                    return
                t = data.strip()
                if t.isdigit():
                    self.cur["text_parts"].append(f" [Abs. {t}] ")
                return
            if self._in_anchor and data.strip().isdigit():
                return  # stray footnote-ref digit outside a sup
            self.cur["text_parts"].append(data)

    def _close_article(self):
        art = self.cur
        self.cur = None
        self.depth = 0
        text = "".join(art.pop("text_parts"))
        # strip the leading "Art. N ..." heading echo and normalize
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^Art\.\s*\d+[a-z]?\s*", "", text)
        art["text"] = text
        self.articles.append(art)


def extract(html: str) -> list[dict]:
    p = ORParser()
    p.feed(html)
    # de-duplicate by article id (keep first, which is the canonical one)
    seen = set()
    out = []
    for a in p.articles:
        if a["article"] in seen:
            continue
        seen.add(a["article"])
        out.append(a)
    return out


def main(langs: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("prsg_*.html"))
    if langs:
        files = [f for f in files if f.stem.split("_")[1] in langs]
    for f in files:
        lang = f.stem.split("_")[1]
        chunks = extract(f.read_text(encoding="utf-8"))
        empty = sum(1 for c in chunks if not c["text"])
        (OUT / f"prsg_{lang}.json").write_text(
            json.dumps({"lang": lang, "law": "PrSG", "chunks": chunks}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"  prsg_{lang}: {len(chunks)} articles ({empty} empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

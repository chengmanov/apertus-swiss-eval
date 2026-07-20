"""Extract Rz-chunked text from FINMA circular PDFs.

FINMA circulars number every normative paragraph with a margin note
(Randziffer, "Rz") -- the citation currency of Swiss financial supervision and
the anchor that lets this eval score citation faithfulness deterministically.

The catch: in current FINMA PDF exports the *unamended* margin numbers are tiny
vector images, not text. Only amended markers ("6.1*", "13.2*") and bare
asterisks decode as text. We therefore reconstruct the numbering geometrically:

  1. Collect marker rows in the right margin band (x0 > MARGIN_X): digit
     images (height ~6pt) and/or anchor text matching \\d+(\\.\\d+)?\\*?.
  2. Sort rows by (page, top) and assign numbers sequentially from 1.
     Text anchors override the counter (after "6" an anchor "6.1" pins the
     row; the next image row then continues at "7").
  3. Self-check: a digit image's width encodes its digit count
     (~4pt per digit). A mismatch between assigned number and observed digit
     count is reported as a warning -- every warning gets a manual review.
  4. Body text (x0 < MARGIN_X, body font size) is sliced into chunks by marker
     row positions; section headings (I., II., A., ...) become chunk metadata;
     footnote-sized text is excluded.

Output: data/text/<slug>_<lang>.json
  {"slug", "lang", "chunks": [{"rz", "section", "text"}...], "warnings": [...]}

Usage: uv run python scripts/extract_text.py [name-substring ...]
"""

import json
import re
import statistics
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "text"

MARGIN_X = 508          # left edge of the Rz marker band
HEADER_Y = 60           # ignore everything above (logo, header)
FOOTER_Y = 790          # ignore everything below (page number)
ANCHOR = re.compile(r"^(\d{1,3}(?:\.\d{1,2})?)\*?$")
HEADING = re.compile(r"^\s*(?:[IVXLC]+|[A-Z]|[a-z]{1,2})[.)]\s+\S")
ROW_TOL = 4             # px tolerance when clustering into rows/lines
AMENDMENT_LIST = re.compile(
    r"(Verzeichnis der Änderungen|List of amendments|Liste des modifications|Elenco delle modifiche)", re.I)


def cluster(items: list[dict], key: str, tol: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for item in sorted(items, key=lambda i: i[key]):
        if groups and abs(item[key] - groups[-1][0][key]) <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def next_integer(current: str | None) -> str:
    if current is None:
        return "1"
    return str(int(float(current)) + 1)


def extract(pdf_path: Path) -> dict:
    markers: list[dict] = []   # {page, top, anchor: str|None, img_digits: int|None}
    body_lines: list[dict] = []  # {page, top, text}
    warnings: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        # Content starts at the first page carrying a genuine margin marker:
        # a digit image, or an amended/inserted text anchor ("5*", "6.1").
        # This skips title and table-of-contents pages, whose right-aligned
        # page/Rz references would otherwise be misread as anchors.
        start_page = 0
        for pageno, page in enumerate(pdf.pages):
            has_img = any(i["x0"] > MARGIN_X and i["height"] < 10 and HEADER_Y < i["top"] < FOOTER_Y
                          for i in page.images)
            has_amended = any(w["x0"] > MARGIN_X and re.match(r"^\d+(\.\d+)?\*$|^\d+\.\d+$", w["text"])
                              for w in page.extract_words())
            if has_img or has_amended:
                start_page = pageno
                break

        reached_amendment_list = False

        for pageno, page in enumerate(pdf.pages):
            if pageno < start_page or reached_amendment_list:
                continue
            words = page.extract_words(extra_attrs=["size"])

            # The amendment history ("Verzeichnis der Änderungen") ends the
            # normative content. Process only content above its heading, then stop.
            cutoff_top = FOOTER_Y
            probe = " ".join(w["text"] for w in sorted(words, key=lambda w: (w["top"], w["x0"])))
            if AMENDMENT_LIST.search(probe):
                heads = [w["top"] for w in words if AMENDMENT_LIST.search(w["text"])
                         or w["text"] in ("Verzeichnis", "Liste", "Elenco")]
                # fall back to word-pair detection ("List of amendments" spans words)
                if not heads:
                    heads = [w["top"] for w in words if w["text"] in ("amendments", "Änderungen", "modifications", "modifiche")]
                if heads:
                    cutoff_top = min(heads) - 2
                    reached_amendment_list = True
            words = [w for w in words if w["top"] < cutoff_top]

            body_sizes = [w["size"] for w in words if w["x0"] < MARGIN_X and HEADER_Y < w["top"] < FOOTER_Y]
            main_size = statistics.median(body_sizes) if body_sizes else 10.0

            # --- marker rows: anchor text in the margin band
            band_items = []
            for w in words:
                if w["x0"] > MARGIN_X and HEADER_Y < w["top"] < FOOTER_Y:
                    m = ANCHOR.match(w["text"])
                    if m:
                        band_items.append({"top": w["top"], "anchor": m.group(1), "img_digits": None})
            # --- marker rows: digit images in the margin band
            for img in page.images:
                if img["x0"] > MARGIN_X and HEADER_Y < img["top"] < cutoff_top and img["height"] < 10:
                    digits = max(1, round(img["width"] / 4))
                    band_items.append({"top": img["top"], "anchor": None, "img_digits": digits})

            for row in cluster(band_items, "top", ROW_TOL):
                anchor = next((r["anchor"] for r in row if r["anchor"]), None)
                img_digits = next((r["img_digits"] for r in row if r["img_digits"]), None)
                markers.append({"page": pageno, "top": row[0]["top"], "anchor": anchor, "img_digits": img_digits})

            # --- body lines (main font only; drops footnotes and page furniture)
            body_words = [w for w in words
                          if w["x0"] < MARGIN_X and HEADER_Y < w["top"] < FOOTER_Y
                          and w["size"] > main_size - 1.2]
            for line_words in cluster(body_words, "top", 2.5):
                text = " ".join(w["text"] for w in sorted(line_words, key=lambda w: w["x0"]))
                body_lines.append({"page": pageno, "top": line_words[0]["top"], "text": text})

    markers.sort(key=lambda m: (m["page"], m["top"]))
    body_lines.sort(key=lambda l: (l["page"], l["top"]))

    # --- assign Rz numbers to marker rows
    current: str | None = None
    for m in markers:
        if m["anchor"]:
            m["rz"] = m["anchor"]
        else:
            m["rz"] = next_integer(current)
            if m["img_digits"] is not None:
                assigned_digits = len(m["rz"].replace(".", ""))
                if assigned_digits != m["img_digits"]:
                    warnings.append(
                        f"digit-count mismatch p{m['page']+1} top{m['top']:.0f}: "
                        f"assigned Rz {m['rz']} ({assigned_digits} digits), image suggests {m['img_digits']}")
        current = m["rz"]

    # --- slice body lines into chunks
    def pos(line_or_marker) -> tuple:
        return (line_or_marker["page"], line_or_marker["top"])

    chunks: list[dict] = []
    preamble: list[str] = []
    section = ""
    idx = -1  # current marker index; -1 = before first Rz
    marker_positions = [pos(m) for m in markers]
    buffers: list[list[str]] = [[] for _ in markers]
    pending_heading: list[str] = []

    for line in body_lines:
        while idx + 1 < len(markers) and pos(line) >= (marker_positions[idx + 1][0], marker_positions[idx + 1][1] - ROW_TOL):
            idx += 1
        if HEADING.match(line["text"]) and len(line["text"]) < 80:
            pending_heading.append(line["text"].strip())
            continue
        if idx < 0:
            preamble.append(line["text"])
            continue
        if pending_heading and not buffers[idx]:
            # heading directly above this Rz belongs to it as section context
            markers[idx]["section"] = " / ".join(pending_heading)
            pending_heading = []
        elif pending_heading:
            markers[idx].setdefault("trailing_heading", " / ".join(pending_heading))
            pending_heading = []
        buffers[idx].append(line["text"])

    # headings observed *after* a chunk's text belong to the next chunk
    for i, m in enumerate(markers):
        if "trailing_heading" in m and i + 1 < len(markers) and "section" not in markers[i + 1]:
            markers[i + 1]["section"] = m.pop("trailing_heading")
        m.pop("trailing_heading", None)

    for m, buf in zip(markers, buffers):
        if "section" in m:
            section = m["section"]
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        text = re.sub(r"([a-zäöüéè])- ([a-zäöüéè])", r"\1\2", text)  # de-hyphenate line breaks
        chunks.append({"rz": m["rz"], "section": section, "text": text})

    empty = [c["rz"] for c in chunks if not c["text"]]
    if empty:
        warnings.append(f"empty chunks for Rz: {', '.join(empty[:10])}")

    return {"chunks": chunks, "preamble": re.sub(r"\s+", " ", " ".join(preamble))[:2000], "warnings": warnings}


def main(patterns: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    if patterns:
        pdfs = [p for p in pdfs if any(pat in p.name for pat in patterns)]
    for pdf in pdfs:
        slug, _, lang = pdf.stem.rpartition("_")
        result = extract(pdf)
        result.update({"slug": slug, "lang": lang})
        dest = OUT / f"{pdf.stem}.json"
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        n = len(result["chunks"])
        last = result["chunks"][-1]["rz"] if n else "-"
        w = len(result["warnings"])
        print(f"  {pdf.stem}: {n} Rz chunks (last Rz {last}), {w} warnings")
        for warning in result["warnings"][:4]:
            print(f"      ! {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

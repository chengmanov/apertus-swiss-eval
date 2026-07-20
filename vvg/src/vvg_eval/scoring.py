"""Deterministic scoring: answer accuracy + article-citation faithfulness.

No LLM judge. Answer scoring depends on the item's answer_type; citation
scoring is exact article-number matching against gold_article (plus explicitly
listed also_acceptable_article). The response is parsed robustly (answer +
citation extracted wherever they appear); strict two-line format compliance is
recorded and reported separately, not used to zero out domain-correct answers.
"""

import re
import unicodedata
from dataclasses import dataclass

ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+)", re.I)
SOURCE_LINE = re.compile(r"SOURCE\s*:\s*(.+)", re.I)
ARTICLES = re.compile(r"\bArt\.?\s*([0-9]+[a-z]?)", re.I)
# Start of a citation clause, in the forms the models actually emit.
CITATION_START = re.compile(r"(?:\bSOURCE\s*:|\bQuelle\s*:|\bVVG\b\s*Art|\bLCA\b\s*Art|\bArt\.)", re.I)
STRICT_FORMAT = re.compile(r"^\s*ANSWER\s*:\s*.+\n\s*SOURCE\s*:\s*.+", re.I)

YES = {"yes", "ja", "oui", "si"}
NO = {"no", "nein", "non"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\"'`«»„“”().,;:!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_number(text: str) -> str | None:
    m = re.search(r"-?\d+(?:[.,']\d+)*", text.replace("'", ""))
    if not m:
        return None
    return m.group(0).replace(",", ".").rstrip(".0") or "0"


@dataclass
class Parsed:
    ok: bool
    answer: str = ""
    articles: tuple[str, ...] = ()
    strict_format: bool = False


def parse_response(raw: str) -> Parsed:
    raw = raw or ""
    strict = bool(STRICT_FORMAT.match(raw.strip()))
    articles = tuple(dict.fromkeys(a.lower() for a in ARTICLES.findall(raw)))

    am = ANSWER_LINE.search(raw)
    if am:
        answer = CITATION_START.split(am.group(1))[0]
    else:
        head = CITATION_START.split(raw, maxsplit=1)[0]
        answer = head.strip().splitlines()[0] if head.strip() else ""
    answer = answer.strip().strip(";[(- \t").strip()

    return Parsed(ok=bool(answer) or bool(articles), answer=answer,
                  articles=articles, strict_format=strict)


def _match_one(candidate: str, gold: str, accept: list[str], answer_type: str) -> bool:
    cand_n = normalize(candidate)
    variants = [normalize(v) for v in [gold, *accept]]
    if answer_type == "yesno":
        tokens = set(cand_n.split())
        gold_yes = normalize(gold) in YES
        has_yes, has_no = tokens & YES, tokens & NO
        if has_yes and has_no:
            return False
        return bool(has_yes) if gold_yes else bool(has_no)
    if answer_type in ("number", "duration"):
        cn = norm_number(cand_n)
        if cn is None:
            return any(cand_n == v or v in cand_n for v in variants if v)
        if any(cn == norm_number(v) for v in variants if norm_number(v) is not None):
            if answer_type == "duration":
                unit_words = {w for v in variants for w in v.split() if not re.match(r"^-?[\d.,]+$", w)}
                if unit_words:
                    return bool(unit_words & set(cand_n.split()))
            return True
        return False
    return any(cand_n == v or v in cand_n or cand_n in v for v in variants if v)


def score_answer(parsed: Parsed, item: dict) -> bool:
    if not parsed.ok:
        return False
    if item["answer_type"] == "list":
        golds: list[str] = item["gold"]
        accepts: list[list[str]] = item.get("accept") or [[] for _ in golds]
        parts = [p for p in re.split(r"[;\n]|,(?=\s)", parsed.answer) if p.strip()]
        matched, used = 0, set()
        for g_idx, g in enumerate(golds):
            for p_idx, p in enumerate(parts):
                if p_idx in used:
                    continue
                if _match_one(p, g, accepts[g_idx] if g_idx < len(accepts) else [], "short"):
                    matched += 1
                    used.add(p_idx)
                    break
        return matched == len(golds) and len(parts) <= len(golds) + 1
    return _match_one(parsed.answer, item["gold"], item.get("accept", []), item["answer_type"])


def score_citation(parsed: Parsed, item: dict) -> bool:
    if not parsed.ok or not parsed.articles:
        return False
    acceptable = {a.lower() for a in item["gold_article"]} | {a.lower() for a in item.get("also_acceptable_article", [])}
    return any(a in acceptable for a in parsed.articles)

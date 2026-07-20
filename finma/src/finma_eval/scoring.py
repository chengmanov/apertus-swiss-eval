"""Deterministic scoring: answer accuracy + citation faithfulness.

No LLM judge anywhere. Answer scoring depends on the item's answer_type;
citation scoring is exact Rz matching against gold_rz (plus explicitly listed
also_acceptable_rz). A response that cannot be parsed under the two-line
output contract scores zero on both dimensions -- format discipline is part
of the measured task.
"""

import re
import unicodedata
from dataclasses import dataclass

ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+)", re.I)
SOURCE_LINE = re.compile(r"SOURCE\s*:\s*(.+)", re.I)
RZ_NUMBERS = re.compile(r"Rz\.?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
CIRCULAR = re.compile(r"(\d{4})\s*/\s*0?(\d{1,2})")
# Start of a citation clause, in any of the forms the models actually emit:
# "SOURCE:", "Quelle:", "[FINMA-RS ...", "(FINMA ...", or a bare "FINMA-RS ...".
CITATION_START = re.compile(r"(?:\bSOURCE\s*:|\bQuelle\s*:|[\[(]?\s*FINMA[- ]?RS\b)", re.I)
STRICT_FORMAT = re.compile(r"^\s*ANSWER\s*:\s*.+\n\s*SOURCE\s*:\s*.+", re.I)

YES = {"yes", "ja", "oui", "si"}
NO = {"no", "nein", "non"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\"'`«»„“”().,;:!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm_number(text: str) -> str | None:
    m = re.search(r"-?\d+(?:[.,']\d+)*", text.replace("'", ""))
    if not m:
        return None
    return m.group(0).replace(",", ".").rstrip(".0") or "0"


@dataclass
class Parsed:
    ok: bool
    answer: str = ""
    circular: str | None = None
    rz: tuple[str, ...] = ()
    strict_format: bool = False


def parse_response(raw: str) -> Parsed:
    """Robustly extract answer + citation from a model response.

    The system prompt asks for a two-line ANSWER:/SOURCE: format, but small
    models frequently answer correctly in other shapes ("nein; [FINMA-RS
    2010/01 Rz 77]", "5 Jahre\\nSOURCE: ..."). Scoring the domain task should
    not hinge on the colon, so we parse the *content* wherever it appears and
    record strict-format compliance separately (reported, not scored).
    """
    raw = raw or ""
    strict = bool(STRICT_FORMAT.match(raw.strip()))

    # Citation: search the whole response for a circular id and Rz numbers.
    circular = None
    cm = CIRCULAR.search(raw)
    if cm:
        circular = f"{cm.group(1)}/{int(cm.group(2)):02d}"
    rz = tuple(dict.fromkeys(RZ_NUMBERS.findall(raw)))  # dedup, keep order

    # Answer: prefer an explicit ANSWER: line; else take the text before the
    # first citation clause; else the first non-empty line.
    am = ANSWER_LINE.search(raw)
    if am:
        answer = am.group(1)
        # drop an inline SOURCE/citation that shares the ANSWER line
        answer = CITATION_START.split(answer)[0]
    else:
        head = CITATION_START.split(raw, maxsplit=1)[0]
        answer = head.strip().splitlines()[0] if head.strip() else ""
    answer = answer.strip().strip(";[(- \t").strip()

    ok = bool(answer) or bool(rz)
    return Parsed(ok=ok, answer=answer, circular=circular, rz=rz, strict_format=strict)


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
            # word-form answers ("one third", "fünf Jahre"): fall back to string match
            return any(cand_n == v or v in cand_n for v in variants if v)
        if any(cn == norm_number(v) for v in variants if norm_number(v) is not None):
            if answer_type == "duration":
                # unit must match some variant's unit word if the variants carry one
                unit_words = {w for v in variants for w in v.split() if not re.match(r"^-?[\d.,]+$", w)}
                if unit_words:
                    return bool(unit_words & set(cand_n.split()))
            return True
        return False
    # short: normalized equality or containment either way against any variant
    return any(cand_n == v or v in cand_n or cand_n in v for v in variants if v)


def score_answer(parsed: Parsed, item: dict) -> bool:
    if not parsed.ok:
        return False
    if item["answer_type"] == "list":
        golds: list[str] = item["gold"]
        accepts: list[list[str]] = item.get("accept") or [[] for _ in golds]
        parts = [p for p in re.split(r"[;\n]|,(?=\s)", parsed.answer) if p.strip()]
        matched = 0
        used: set[int] = set()
        for g_idx, g in enumerate(golds):
            for p_idx, p in enumerate(parts):
                if p_idx in used:
                    continue
                if _match_one(p, g, accepts[g_idx] if g_idx < len(accepts) else [], "short"):
                    matched += 1
                    used.add(p_idx)
                    break
        # full credit only: all gold items present, no more than one spurious extra
        return matched == len(golds) and len(parts) <= len(golds) + 1
    return _match_one(parsed.answer, item["gold"], item.get("accept", []), item["answer_type"])


def score_citation(parsed: Parsed, item: dict) -> bool:
    if not parsed.ok or not parsed.rz:
        return False
    if parsed.circular is not None:
        want = item["circular"]
        got = parsed.circular
        if f"{int(want.split('/')[0])}/{int(want.split('/')[1]):02d}" != f"{int(got.split('/')[0])}/{int(got.split('/')[1]):02d}":
            return False
    else:
        return False  # a citation without the circular id is not a usable citation
    acceptable = set(item["gold_rz"]) | set(item.get("also_acceptable_rz", []))
    return any(r in acceptable for r in parsed.rz)

"""Prompt templates for the four evaluation arms.

Output contract (all arms, both languages): exactly two lines --

  ANSWER: <short answer>
  SOURCE: FINMA-RS <circular> Rz <number>[, Rz <number>...]

The contract is stated in the system prompt and enforced nowhere else; parsing
failures are scored as failures (format discipline is part of what a
production system needs from a model, so we measure it).
"""

from .corpus import Chunk

SYSTEM = {
    "en": (
        "You are a Swiss financial-regulation assistant. Answer questions about "
        "FINMA circulars precisely and briefly.\n"
        "Reply with exactly two lines and nothing else:\n"
        "ANSWER: <the short answer -- a yes/no, a number, or a short phrase; for "
        "list questions, items separated by semicolons>\n"
        "SOURCE: FINMA-RS <circular number> Rz <margin number(s)>\n"
        "Example:\nANSWER: no\nSOURCE: FINMA-RS 2018/03 Rz 9"
    ),
    "de": (
        "Du bist ein Assistent für Schweizer Finanzmarktregulierung. Beantworte "
        "Fragen zu FINMA-Rundschreiben präzise und knapp.\n"
        "Antworte mit genau zwei Zeilen und sonst nichts:\n"
        "ANSWER: <die kurze Antwort -- ja/nein, eine Zahl oder eine kurze "
        "Formulierung; bei Listenfragen die Elemente mit Semikolon getrennt>\n"
        "SOURCE: FINMA-RS <Rundschreiben-Nummer> Rz <Randziffer(n)>\n"
        "Beispiel:\nANSWER: nein\nSOURCE: FINMA-RS 2018/03 Rz 9"
    ),
}

CONTEXT_HEADER = {
    "en": "Excerpts from FINMA circulars (cite the margin numbers you rely on):",
    "de": "Auszüge aus FINMA-Rundschreiben (zitiere die Randziffern, auf die du dich stützt):",
}

QUESTION_HEADER = {"en": "Question:", "de": "Frage:"}


def user_prompt(question: str, lang: str, context: list[Chunk] | None = None) -> str:
    parts: list[str] = []
    if context:
        parts.append(CONTEXT_HEADER[lang])
        for c in context:
            parts.append(f"[{c.ref}] ({c.section})\n{c.text}")
        parts.append("")
    parts.append(f"{QUESTION_HEADER[lang]} {question}")
    return "\n\n".join(parts)

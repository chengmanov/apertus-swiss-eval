"""Prompt templates for the four evaluation arms (OR article Q&A).

Output contract (both languages): exactly two lines --

  ANSWER: <short answer>
  SOURCE: OR Art. <number>[, Art. <number>...]

Parsing failures are scored as failures (format discipline is part of what a
production system needs), but the parser extracts answer + citation wherever
they appear and reports strict-format compliance separately.
"""

from .corpus import Chunk

SYSTEM = {
    "de": (
        "Du bist ein Assistent für Schweizer Vertragsrecht. Beantworte Fragen "
        "zum Obligationenrecht (OR) präzise und knapp.\n"
        "Antworte mit genau zwei Zeilen und sonst nichts:\n"
        "ANSWER: <die kurze Antwort -- ja/nein, eine Zahl, eine Frist oder eine "
        "kurze Formulierung; bei Listen die Elemente mit Semikolon getrennt>\n"
        "SOURCE: OR Art. <Artikelnummer(n)>\n"
        "Beispiel:\nANSWER: nein\nSOURCE: OR Art. 335"
    ),
    "fr": (
        "Tu es un assistant en droit suisse des contrats. Réponds aux questions "
        "sur le Code des obligations (CO) de manière précise et concise.\n"
        "Réponds avec exactement deux lignes et rien d'autre :\n"
        "ANSWER: <la réponse courte -- oui/non, un nombre, un délai ou une "
        "formulation brève ; pour les listes, éléments séparés par des points-virgules>\n"
        "SOURCE: OR Art. <numéro(s) d'article>\n"
        "Exemple :\nANSWER: non\nSOURCE: OR Art. 335"
    ),
}

CONTEXT_HEADER = {
    "de": "Auszüge aus dem Obligationenrecht (zitiere die Artikel, auf die du dich stützt):",
    "fr": "Extraits du Code des obligations (cite les articles sur lesquels tu te fondes) :",
}

QUESTION_HEADER = {"de": "Frage:", "fr": "Question :"}


def user_prompt(question: str, lang: str, context: list[Chunk] | None = None) -> str:
    parts: list[str] = []
    if context:
        parts.append(CONTEXT_HEADER[lang])
        for c in context:
            parts.append(f"[{c.ref}] ({c.section})\n{c.text}")
        parts.append("")
    parts.append(f"{QUESTION_HEADER[lang]} {question}")
    return "\n\n".join(parts)

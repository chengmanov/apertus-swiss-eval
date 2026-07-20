"""Prompt templates for the four evaluation arms (DSG article Q&A).

Output contract (both languages): exactly two lines --

  ANSWER: <short answer>
  SOURCE: DSG Art. <number>[, Art. <number>...]   (FR: LPD Art. ...)

Parsing failures are scored as failures (format discipline is part of what a
production system needs), but the parser extracts answer + citation wherever
they appear and reports strict-format compliance separately.
"""

from .corpus import Chunk

# Official abbreviation of the law per language (German DSG / French LPD).
LABEL = {"de": "DSG", "fr": "LPD"}

SYSTEM = {
    "de": (
        "Du bist ein Assistent für das Schweizer Datenschutzrecht. Beantworte "
        "Fragen zum Bundesgesetz über den Datenschutz (DSG) präzise und knapp.\n"
        "Antworte mit genau zwei Zeilen und sonst nichts:\n"
        "ANSWER: <die kurze Antwort -- ja/nein, eine Zahl, eine Frist oder eine "
        "kurze Formulierung; bei Listen die Elemente mit Semikolon getrennt>\n"
        "SOURCE: DSG Art. <Artikelnummer(n)>\n"
        "Beispiel:\nANSWER: nein\nSOURCE: DSG Art. 6"
    ),
    "fr": (
        "Tu es un assistant en droit suisse de la protection des données. Réponds "
        "aux questions sur la loi fédérale sur la protection des données (LPD) de "
        "manière précise et concise.\n"
        "Réponds avec exactement deux lignes et rien d'autre :\n"
        "ANSWER: <la réponse courte -- oui/non, un nombre, un délai ou une "
        "formulation brève ; pour les listes, éléments séparés par des points-virgules>\n"
        "SOURCE: LPD Art. <numéro(s) d'article>\n"
        "Exemple :\nANSWER: non\nSOURCE: LPD Art. 6"
    ),
}

CONTEXT_HEADER = {
    "de": "Auszüge aus dem Datenschutzgesetz (zitiere die Artikel, auf die du dich stützt):",
    "fr": "Extraits de la loi sur la protection des données (cite les articles sur lesquels tu te fondes) :",
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

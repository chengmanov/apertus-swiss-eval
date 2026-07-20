"""Prompt templates for the four evaluation arms (VVG article Q&A).

Output contract (both languages): exactly two lines --

  ANSWER: <short answer>
  SOURCE: VVG Art. <number>[, Art. <number>...]   (FR: LCA Art. ...)
"""

from .corpus import Chunk

# Official abbreviation of the law per language (German VVG / French LCA).
LABEL = {"de": "VVG", "fr": "LCA"}

SYSTEM = {
    "de": (
        "Du bist ein Assistent für das Schweizer Versicherungsvertragsrecht. "
        "Beantworte Fragen zum Bundesgesetz über den Versicherungsvertrag (VVG) "
        "präzise und knapp.\n"
        "Antworte mit genau zwei Zeilen und sonst nichts:\n"
        "ANSWER: <die kurze Antwort -- ja/nein, eine Zahl, eine Frist oder eine "
        "kurze Formulierung; bei Listen die Elemente mit Semikolon getrennt>\n"
        "SOURCE: VVG Art. <Artikelnummer(n)>\n"
        "Beispiel:\nANSWER: nein\nSOURCE: VVG Art. 4"
    ),
    "fr": (
        "Tu es un assistant en droit suisse du contrat d'assurance. Réponds aux "
        "questions sur la loi fédérale sur le contrat d'assurance (LCA) de "
        "manière précise et concise.\n"
        "Réponds avec exactement deux lignes et rien d'autre :\n"
        "ANSWER: <la réponse courte -- oui/non, un nombre, un délai ou une "
        "formulation brève ; pour les listes, éléments séparés par des points-virgules>\n"
        "SOURCE: LCA Art. <numéro(s) d'article>\n"
        "Exemple :\nANSWER: non\nSOURCE: LCA Art. 4"
    ),
}

CONTEXT_HEADER = {
    "de": "Auszüge aus dem Versicherungsvertragsgesetz (zitiere die Artikel, auf die du dich stützt):",
    "fr": "Extraits de la loi sur le contrat d'assurance (cite les articles sur lesquels tu te fondes) :",
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

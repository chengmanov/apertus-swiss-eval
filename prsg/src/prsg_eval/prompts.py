"""Prompt templates for the four evaluation arms (PrSG article Q&A).

Output contract (both languages): exactly two lines --

  ANSWER: <short answer>
  SOURCE: PrSG Art. <number>[, Art. <number>...]   (FR: LSPro Art. ...)
"""

from .corpus import Chunk

# Official abbreviation of the law per language (German PrSG / French LSPro).
LABEL = {"de": "PrSG", "fr": "LSPro"}

SYSTEM = {
    "de": (
        "Du bist ein Assistent für das Schweizer Produktesicherheitsrecht. "
        "Beantworte Fragen zum Bundesgesetz über die Produktesicherheit (PrSG) "
        "präzise und knapp.\n"
        "Antworte mit genau zwei Zeilen und sonst nichts:\n"
        "ANSWER: <die kurze Antwort -- ja/nein, eine Zahl, eine Frist oder eine "
        "kurze Formulierung; bei Listen die Elemente mit Semikolon getrennt>\n"
        "SOURCE: PrSG Art. <Artikelnummer(n)>\n"
        "Beispiel:\nANSWER: ja\nSOURCE: PrSG Art. 3"
    ),
    "fr": (
        "Tu es un assistant en droit suisse de la sécurité des produits. Réponds "
        "aux questions sur la loi fédérale sur la sécurité des produits (LSPro) "
        "de manière précise et concise.\n"
        "Réponds avec exactement deux lignes et rien d'autre :\n"
        "ANSWER: <la réponse courte -- oui/non, un nombre, un délai ou une "
        "formulation brève ; pour les listes, éléments séparés par des points-virgules>\n"
        "SOURCE: LSPro Art. <numéro(s) d'article>\n"
        "Exemple :\nANSWER: oui\nSOURCE: LSPro Art. 3"
    ),
}

CONTEXT_HEADER = {
    "de": "Auszüge aus dem Produktesicherheitsgesetz (zitiere die Artikel, auf die du dich stützt):",
    "fr": "Extraits de la loi sur la sécurité des produits (cite les articles sur lesquels tu te fondes) :",
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

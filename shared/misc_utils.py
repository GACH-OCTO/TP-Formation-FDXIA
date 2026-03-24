"""Utilitaires génériques pour écrire des JSON et parser du texte"""

import json
from pathlib import Path


def write_json_file(file_path: str | Path, data: dict[str, object]) -> None:
    """Écrire un dictionnaire JSON formaté, en créant le dossier parent si nécessaire"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_and_parse_json_from_text(text: str) -> dict[str, object]:
    """TODO extraire un objet JSON depuis une sortie LLM mixte (texte + JSON)

    Entrée
    - text : texte complet retourné par le modèle

    Sortie
    - dictionnaire JSON parsé depuis `text`

    Erreurs
    - ValueError si accolades non équilibrées ou JSON absent
    """
    # TODO : conserver un état de parsing explicite pour ignorer les accolades dans les chaînes
    start_index = -1
    depth = 0
    in_string = False
    is_escaped = False
    candidates: list[tuple[int, int]] = []

    # TODO : collecter les candidats JSON complets puis parser le dernier candidat valide
    for index, char in enumerate(text):
        if in_string:
            if is_escaped:
                is_escaped = False
            elif char == "\\":
                is_escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}":
            if depth == 0:
                raise ValueError("Found closing brace without opening brace")
            depth -= 1
            if depth == 0 and start_index >= 0:
                candidates.append((start_index, index + 1))

    if depth != 0:
        raise ValueError("Unclosed JSON object in text")
    if not candidates:
        raise ValueError("No JSON object found in text")

    last_start, last_end = candidates[-1]
    parsed_data = json.loads(text[last_start:last_end])
    if not isinstance(parsed_data, dict):
        raise ValueError("Parsed JSON is not an object")
    return parsed_data

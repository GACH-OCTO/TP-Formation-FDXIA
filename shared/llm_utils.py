from pydantic import BaseModel, Field
from google.genai import types

from shared.config import genai_client, project_settings


class LLMRequest(BaseModel):
    """TODO définir le format d'entrée d'un appel LLM TP1

    Champs
    - system_prompt : consigne système optionnelle
    - user_prompt : prompt utilisateur obligatoire
    """
    ...


class LLMResponse(BaseModel):
    """TODO définir le format de sortie normalisé de `run_llm`

    Champs
    - output : texte final du modèle
    - input_tokens : nombre de tokens d'entrée
    - output_tokens : nombre de tokens de sortie
    - total_tokens : total des tokens
    - raw_response : réponse brute du provider pour debug
    """
    ...


async def run_llm(request: LLMRequest) -> LLMResponse:
    """TODO exécuter un appel modèle unique réutilisable dans TP1

    Entrée
    - request : `LLMRequest` (`system_prompt`, `user_prompt`)

    Sortie
    - `LLMResponse` avec texte, métriques tokens et réponse brute
    """

    config = types.GenerateContentConfig(
        temperature=...,
        top_p=...,
        top_k=...,
        max_output_tokens=...,
        thinking_config=types.ThinkingConfig(
            thinking_budget=...
        ),
        system_instruction=request.system_prompt,
    )

    response = await genai_client.aio.models.generate_content(
        model=...,
        contents=...,
        config=...,
    )

    usage_metadata = response.usage_metadata

    return ...

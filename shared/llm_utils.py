from pydantic import BaseModel, Field
from google.genai import types

from shared.config import genai_client, project_settings


class LLMRequest(BaseModel):
    """TODO définir le format d'entrée d'un appel LLM TP1

    Champs
    - system_prompt : consigne système optionnelle
    - user_prompt : prompt utilisateur obligatoire
    """
    system_prompt: str
    user_prompt: str


class LLMResponse(BaseModel):
    """TODO définir le format de sortie normalisé de `run_llm`

    Champs
    - output : texte final du modèle
    - input_tokens : nombre de tokens d'entrée
    - output_tokens : nombre de tokens de sortie
    - total_tokens : total des tokens
    - raw_response : réponse brute du provider pour debug
    """
    output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw_response: dict


async def run_llm(request: LLMRequest) -> LLMResponse:
    """TODO exécuter un appel modèle unique réutilisable dans TP1

    Entrée
    - request : `LLMRequest` (`system_prompt`, `user_prompt`)

    Sortie
    - `LLMResponse` avec texte, métriques tokens et réponse brute
    """

    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        ),
        system_instruction=request.system_prompt,
    )

    response = await genai_client.aio.models.generate_content(
        model=project_settings.llm_model_name,
        contents=[request.user_prompt],
        config=config,
    )

    usage_metadata = response.usage_metadata

    return LLMResponse(
        output=response.text,
        input_tokens=usage_metadata.prompt_token_count,
        output_tokens=usage_metadata.candidates_token_count,
        total_tokens=usage_metadata.total_token_count,
        raw_response=response.model_dump(),
    )


import json
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.usage import UsageLimits


TRACE_SEPARATOR = "-" * 80


class EventStreamHandler:
    """Gestionnaire de flux pour journaliser les appels outils en temps réel

    Méthodes
    - `__call__` : consomme les événements et ajoute des blocs formatés
    """

    async def __call__(self, _run_context, event_stream) -> None:
        """Formater les événements d'appel outil et de résultat outil

        Entrées
        - _run_context : contexte d'exécution pydantic-ai
        - event_stream : flux asynchrone des événements modèle/outils
        """
        step = 1
        async for event in event_stream:
            match event:
                case FunctionToolCallEvent():
                    ...
                case FunctionToolResultEvent():
                    ...
                case _:
                    continue
            ...
            step += 1


async def run_agent_realtime_logging(
    agent: Agent,
    prompt: str,
    log_path: Path,
    max_steps: int = 12,
) -> AgentRunResult:
    """TODO Exécuter un agent avec affichage temps réel tronqué et log brut complet

    Entrées
    - agent : instance d'agent pydantic-ai configurée
    - prompt : prompt utilisateur envoyé à `agent.run`
    - log_path : chemin du fichier de log (réponse brute complète JSON)
    - max_steps : nombre maximal d'étapes autorisées

    Sortie
    - `AgentRunResult`
    """
    print("RUN TRACE")
    print(f"- prompt : {prompt}")
    print("=" * 80)

    run_result = await agent.run(
        ...,
        event_stream_handler=...,
        usage_limits=...,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    ...

    return run_result

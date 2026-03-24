from datetime import date

import googlemaps
import requests
from googlemaps.exceptions import ApiError, TransportError
from pydantic import BaseModel, ConfigDict
from tavily import TavilyClient

from shared.config import ROOT_DIR, project_settings
from shared.rag_utils import RAGAssistant


REQUEST_TIMEOUT_SECONDS = 45
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
RAG_VECTOR_DB_DIR = ROOT_DIR / "TP2_travel_planner_RAG" / "data" / "chroma_db_rag_v2"
RAG_ASSISTANT: RAGAssistant | None = None


def get_rag_assistant() -> RAGAssistant:
    """Chargement paresseux de l'assistant RAG
    Instancié à la première utilisation pour éviter une erreur si la base vectorielle n'existe pas encore
    """
    global RAG_ASSISTANT
    if RAG_ASSISTANT is None:
        RAG_ASSISTANT = RAGAssistant(persist_dir=RAG_VECTOR_DB_DIR)
    return RAG_ASSISTANT


class Place(BaseModel):
    """TODO format de lieu normalisé utilisé par les outils agent

    Champs
    - name : libellé lisible du lieu
    - latitude : latitude WGS84
    - longitude : longitude WGS84
    """
    ...



def tool_get_current_date() -> dict[str, str]:
    """TODO retourner la date courante pour ancrer les dates relatives

    Sortie
    - dictionnaire avec
      - current_date : date ISO (`YYYY-MM-DD`)
      - weekday_name : nom du jour
    """
    ...
    return {...}


def tool_geocode_location(query: str) -> dict[str, object]:
    """TODO convertir un lieu texte en coordonnées (utiliser Google Maps Geocoding API)

    Entrées
    - query : lieu au format texte libre

    Sortie
    - dictionnaire `Place` avec `name`, `latitude`, `longitude`
    """
    ...
    place = Place(...)
    return place.model_dump()


def tool_get_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> dict[str, object]:
    """TODO récupérer une prévision météo sur une plage de dates (utiliser Open-Meteo API)

    Entrées
    - latitude : latitude du lieu
    - longitude : longitude du lieu
    - start_date : date de début incluse au format ISO
    - end_date : date de fin incluse au format ISO

    Sortie
    - dictionnaire avec
      - timezone : fuseau de la réponse
      - days : liste journalière (min/max température, précipitations)
    """
    ...
    response = requests.get(
        OPEN_METEO_URL,
        params={
            ...
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    ...
    days: list[dict[str, object]] = []
    for index, day_date in ...:
        days.append(
            {
                "date": ...,
                "min_temp_c": ...,
                "max_temp_c": ...,
                "precipitation_mm": ...,
            }
        )

    return {"timezone": ..., "days": days}


def tool_search_nearby(
    latitude: float,
    longitude: float,
    place_type: str = "restaurant",
    keyword: str = "",
    radius_m: int = 1500,
    limit: int = 8,
) -> dict[str, object]:
    """TODO rechercher des lieux proches à partir de coordonnées (Utiliser Google Maps Places API)

    Entrées
    - latitude : latitude de référence
    - longitude : longitude de référence
    - place_type : type Google Places (`restaurant`, `museum`, ...)
    - keyword : filtre texte optionnel
    - radius_m : rayon de recherche en mètres
    - limit : nombre maximal de lieux retournés

    Sortie
    - dictionnaire avec
      - count : nombre de lieux retenus
      - items : liste sérialisée de `Place`
    """
    ...
    return {
        "count": ...,
        "items": [...],
    }


def tool_retrieve_docs(query: str, top_k: int = 5) -> list[dict[str, object]]:
    """TODO récupérer les chunks RAG internes pour une requête

    Entrées
    - query : texte de recherche
    - top_k : nombre de chunks à retourner

    Sortie
    - liste de dictionnaires avec `source`, `chunk_id`, `text`, `score`
    """
    rag_assistant = get_rag_assistant()
    chunks_with_scores = ...
    ...
    return [
        {
            "source": ...,
            "chunk_id": ...,
            "text": ...,
            "score": ...,
        }
        for chunk, score in chunks_with_scores
    ]


def web_search(query: str, max_results: int = 5) -> dict[str, object]:
    """TODO rechercher des sources web externes (Utiliser Tavily Search API)

    Entrées
    - query : requête de recherche
    - max_results : nombre maximal de résultats

    Sortie
    - dictionnaire avec `count` et `items` (`title`, `link`, `snippet`)
    """
    tavily_client = ...
    response_data = ...
    ...
    return {
        "count": ...,
        "items": ...,
    }


def web_extract(urls: str | list[str], query: str = "") -> dict[str, object]:
    """TODO extraire du contenu web depuis une ou plusieurs URL (Utiliser Tavily Extract API)

    Entrées
    - urls : une URL ou une liste d'URL
    - query : filtre optionnel de focalisation

    Sortie
    - dictionnaire avec `count` et `items` contenant `url` et `content`
    """
    tavily_client = ...
    response_data = ...
    ...
    return {
        "count": ...,
        "items": ...,
    }

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
    model_config = ConfigDict(extra="forbid")
    name: str
    latitude: float
    longitude: float


def tool_get_current_date() -> dict[str, str]:
    """TODO retourner la date courante pour ancrer les dates relatives

    Sortie
    - dictionnaire avec
      - current_date : date ISO (`YYYY-MM-DD`)
      - weekday_name : nom du jour
    """
    current_date = date.today()
    return {
        "current_date": current_date.isoformat(),
        "weekday_name": current_date.strftime("%A"),
    }


def tool_geocode_location(query: str) -> dict[str, object]:
    """TODO convertir un lieu texte en coordonnées (utiliser Google Maps Geocoding API)

    Entrées
    - query : lieu au format texte libre

    Sortie
    - dictionnaire `Place` avec `name`, `latitude`, `longitude`
    """
    # TODO : garder un message d'erreur clair en cas d'échec API réseau
    try:
        geocode_results = googlemaps.Client(key=project_settings.google_api_key).geocode(address=query)
    except (ApiError, TransportError) as error:
        raise ValueError(
            "Échec de la requête Google Maps Geocode. Vérifier la clé API, l'accès réseau et les APIs Maps activées dans GCP."
        ) from error

    first = geocode_results[0]
    location = first["geometry"]["location"]
    place = Place(
        name=str(first["formatted_address"]),
        latitude=float(location["lat"]),
        longitude=float(location["lng"]),
    )
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
    # TODO : valider les dates avant l'appel météo
    parsed_start_date = date.fromisoformat(start_date)
    parsed_end_date = date.fromisoformat(end_date)

    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "start_date": parsed_start_date.isoformat(),
            "end_date": parsed_end_date.isoformat(),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    response_data = response.json()
    daily = response_data["daily"]

    days: list[dict[str, object]] = []
    for index, day_date in enumerate(daily["time"]):
        days.append(
            {
                "date": str(day_date),
                "min_temp_c": daily["temperature_2m_min"][index],
                "max_temp_c": daily["temperature_2m_max"][index],
                "precipitation_mm": daily["precipitation_sum"][index],
            }
        )

    return {"timezone": str(response_data["timezone"]), "days": days}


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
    # TODO : normaliser les filtres avant l'appel API
    normalized_place_type = place_type.strip().lower()
    normalized_keyword = keyword.strip()
    search_parameters: dict[str, object] = {
        "location": (latitude, longitude),
        "radius": radius_m,
        "type": normalized_place_type,
        "keyword": normalized_keyword,
    }

    try:
        response_data = googlemaps.Client(key=project_settings.google_api_key).places_nearby(**search_parameters)
    except (ApiError, TransportError) as error:
        raise ValueError(
            "Échec de la requête Google Maps Nearby. Vérifier la clé API, l'accès réseau et les APIs Maps activées dans GCP."
        ) from error

    results = response_data["results"]

    places: list[Place] = []
    for result in results[:limit]:
        location = result["geometry"]["location"]
        places.append(
            Place(
                name=str(result["name"]),
                latitude=float(location["lat"]),
                longitude=float(location["lng"]),
            )
        )

    return {
        "count": len(places),
        "items": [place.model_dump() for place in places],
    }


def tool_retrieve_docs(query: str, top_k: int = 5) -> list[dict[str, object]]:
    """TODO récupérer les chunks RAG internes pour une requête

    Entrées
    - query : texte de recherche
    - top_k : nombre de chunks à retourner

    Sortie
    - liste de dictionnaires avec `source`, `chunk_id`, `text`, `score`
    """
    # TODO : garder un format simple pour les citations dans la réponse finale
    rag_assistant = get_rag_assistant()
    chunks_with_scores = rag_assistant.search(query=query, top_k=top_k)
    return [
        {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "score": score,
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
    # TODO : conserver des clés stables pour les citations dans le prompt
    tavily_client = TavilyClient(api_key=project_settings.tavily_api_key)
    response_data = tavily_client.search(query=query, max_results=max_results)
    results = response_data.get("results", [])
    items = [
        {
            "title": str(result.get("title", "")),
            "link": str(result.get("url", "")),
            "snippet": str(result.get("content", "")),
        }
        for result in results
    ]
    return {
        "count": len(items),
        "items": items,
    }


def web_extract(urls: str | list[str], query: str = "") -> dict[str, object]:
    """TODO extraire du contenu web depuis une ou plusieurs URL (Utiliser Tavily Extract API)

    Entrées
    - urls : une URL ou une liste d'URL
    - query : filtre optionnel de focalisation

    Sortie
    - dictionnaire avec `count` et `items` contenant `url` et `content`
    """
    # TODO : limiter le contenu retourné pour garder une sortie compacte
    tavily_client = TavilyClient(api_key=project_settings.tavily_api_key)
    response_data = tavily_client.extract(urls=urls, query=query.strip())
    results = response_data.get("results", [])

    items = []
    for result in results:
        raw_content = str(result.get("raw_content", ""))
        items.append(
            {
                "url": str(result.get("url", "")),
                "content": raw_content[:100],
            }
        )

    return {
        "count": len(items),
        "items": items,
    }

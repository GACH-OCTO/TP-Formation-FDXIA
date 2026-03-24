import json
from datetime import date, timedelta

from shared.agent_tools import (
    tool_get_current_date,
    tool_geocode_location,
    tool_get_weather,
    tool_retrieve_docs,
    tool_search_nearby,
    web_extract,
    web_search,
)


REFERENCE_ADDRESS_QUERY = "10 Rue de la Paix, 75002 Paris, France"
REFERENCE_RAG_QUERY = "Rome hidden gems restaurants itinerary budget"
REFERENCE_WEB_SEARCH_QUERY = "best period paris new york travel prices"
REFERENCE_WEB_EXTRACT_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
SECTION_SEPARATOR = "=" * 80


def format_json(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def print_tool_result(tool_name: str, tool_input: dict[str, object], tool_output: dict[str, object]) -> None:
    print(SECTION_SEPARATOR)
    print(f"TOOL : {tool_name}")
    print("- INPUT")
    print(format_json(tool_input))
    print("- OUTPUT")
    print(format_json(tool_output))
    print(SECTION_SEPARATOR)


def test_tool_get_current_date() -> None:
    tool_name = "tool_get_current_date"
    tool_input: dict[str, object] = {}
    tool_output = tool_get_current_date()
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_tool_geocode_location() -> None:
    tool_name = "tool_geocode_location"
    tool_input = {"query": REFERENCE_ADDRESS_QUERY}
    tool_output = tool_geocode_location(query=REFERENCE_ADDRESS_QUERY)
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_tool_get_weather() -> None:
    geocode_output = tool_geocode_location(query=REFERENCE_ADDRESS_QUERY)
    latitude = float(geocode_output["latitude"])
    longitude = float(geocode_output["longitude"])
    start_date = (date.today() + timedelta(days=1)).isoformat()
    end_date = (date.today() + timedelta(days=3)).isoformat()

    tool_name = "tool_get_weather"
    tool_input = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
    }
    tool_output = tool_get_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_tool_search_nearby() -> None:
    geocode_output = tool_geocode_location(query=REFERENCE_ADDRESS_QUERY)
    latitude = float(geocode_output["latitude"])
    longitude = float(geocode_output["longitude"])

    tool_name = "tool_search_nearby"
    tool_input = {
        "latitude": latitude,
        "longitude": longitude,
        "place_type": "restaurant",
        "keyword": "",
        "radius_m": 1200,
        "limit": 3,
    }
    tool_output = tool_search_nearby(
        latitude=latitude,
        longitude=longitude,
        place_type="restaurant",
        keyword="",
        radius_m=1200,
        limit=3,
    )
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_web_search() -> None:
    tool_name = "web_search"
    tool_input = {"query": REFERENCE_WEB_SEARCH_QUERY, "max_results": 3}
    tool_output = web_search(query=REFERENCE_WEB_SEARCH_QUERY, max_results=3)
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_web_extract() -> None:
    tool_name = "web_extract"
    tool_input = {"urls": REFERENCE_WEB_EXTRACT_URL}
    tool_output = web_extract(urls=REFERENCE_WEB_EXTRACT_URL)
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def test_tool_retrieve_docs() -> None:
    tool_name = "tool_retrieve_docs"
    tool_input = {"query": REFERENCE_RAG_QUERY, "top_k": 3}
    tool_output = {
        "items": tool_retrieve_docs(query=REFERENCE_RAG_QUERY, top_k=3),
    }
    print_tool_result(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)


def run_all_tool_tests() -> None:
    test_tool_get_current_date()
    test_tool_get_weather()
    test_tool_retrieve_docs()
    test_tool_geocode_location()
    test_tool_search_nearby()
    test_web_search()
    test_web_extract()


run_all_tool_tests()

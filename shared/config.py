"""
Authentication:
Set `GOOGLE_API_KEY`, `GOOGLE_PROJECT_ID`, and `TAVILY_API_KEY` in `.env` at project root
"""

import os
from pathlib import Path

from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)
google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
google_project_id = os.getenv("GOOGLE_PROJECT_ID", "").strip()
tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()


class ProjectSettings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    # API key for Gemini and Google Maps tools
    google_api_key: str = Field(default=google_api_key, min_length=1)
    google_project_id: str = Field(default=google_project_id, min_length=1)
    tavily_api_key: str = Field(default=tavily_api_key, min_length=1)

    # For LLM and Agent
    llm_model_name: str = "gemini-2.5-flash-lite"
    llm_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    llm_top_k: int = Field(default=40, ge=1)
    llm_max_output_tokens: int = Field(default=5000, ge=512)
    llm_thinking_budget: int = Field(default=1000, ge=0)

    # For RAG (Retrieval Augmented Generation)
    rag_embedding_model_name: str = "gemini-embedding-001"


project_settings = ProjectSettings()

genai_client = genai.Client(
    api_key=project_settings.google_api_key,
)

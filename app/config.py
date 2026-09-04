"""Configuration du projet via variables d'environnement et fichier .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Charge automatiquement les variables depuis le fichier .env à la racine du projet.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_PATH = DATA_DIR / "events_sample.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "metadata.json"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K = int(os.getenv("TOP_K", "3"))
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Paris")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))

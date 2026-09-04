"""Chargement du dataset d'évaluation du système RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset

from app.config import DEFAULT_CITY


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation.json"


def load_evaluation_rows() -> list[dict[str, Any]]:
    """Charge les exemples d'évaluation et remplace le placeholder de ville."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset d'évaluation introuvable : {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Le dataset d'évaluation doit contenir une liste d'objets JSON."
        )

    # Remplace {DEFAULT_CITY} dans toutes les chaînes du dataset.
    for row in data:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = value.replace("{DEFAULT_CITY}", DEFAULT_CITY)

    return data

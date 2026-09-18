"""Ajoute les contextes associés aux `reference_doc_ids` dans un dataset JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_PATH = Path("data/evaluation/rag_evaluation.json")
DEFAULT_METADATA_PATH = Path("data/metadata.json")


def _clean_text(value: Any) -> str:
    """Normalise un texte brut avant de l'utiliser comme contexte."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def load_json(path: Path) -> Any:
    """Charge un fichier JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_metadata_map(metadata: Any) -> dict[str, str]:
    """Construit un mapping id -> texte, en supportant plusieurs formats."""
    mapping: dict[str, str] = {}

    if isinstance(metadata, dict):
        for doc_id, value in metadata.items():
            if isinstance(value, str):
                text = _clean_text(value)
            elif isinstance(value, dict):
                text = ""
                for key in ("page_content", "content", "text", "body"):
                    candidate = value.get(key)
                    if isinstance(candidate, str):
                        text = _clean_text(candidate)
                        break
            else:
                continue

            if text:
                mapping[str(doc_id)] = text
        return mapping

    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue

            doc_id = next(
                (
                    item.get(key)
                    for key in ("uid", "id", "_id", "document_id", "doc_id")
                    if key in item
                ),
                None,
            )

            if doc_id is None:
                continue

            text = ""
            for key in ("page_content", "content", "text", "body"):
                candidate = item.get(key)
                if isinstance(candidate, str):
                    text = _clean_text(candidate)
                    break

            if text:
                mapping[str(doc_id)] = text

    return mapping


def enrich_reference_contexts(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> list[dict[str, Any]]:
    """Ajoute `reference_contexts` pour chaque ligne qui a des `reference_doc_ids`."""
    dataset_file = Path(dataset_path)
    metadata_file = Path(metadata_path)

    dataset = load_json(dataset_file)
    if not isinstance(dataset, list):
        raise ValueError("Le dataset d'évaluation doit être une liste JSON.")

    metadata = load_json(metadata_file)
    metadata_map = build_metadata_map(metadata)

    updated_rows: list[dict[str, Any]] = []

    for row in dataset:
        if not isinstance(row, dict):
            updated_rows.append(row)
            continue

        updated_row = dict(row)
        doc_ids = updated_row.get("reference_doc_ids")

        if isinstance(doc_ids, list) and doc_ids:
            contexts: list[str] = []
            for doc_id in doc_ids:
                if doc_id in (None, ""):
                    continue
                doc_text = metadata_map.get(str(doc_id))
                if doc_text:
                    contexts.append(doc_text)
            if contexts:
                updated_row["reference_contexts"] = contexts
            elif "reference_contexts" not in updated_row:
                updated_row["reference_contexts"] = []

        updated_rows.append(updated_row)

    return updated_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remplir reference_contexts à partir des reference_doc_ids."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Chemin vers le dataset JSON contenant reference_doc_ids.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Chemin vers le fichier metadata.json contenant les textes associées aux ids.",
    )
    args = parser.parse_args()

    updated_rows = enrich_reference_contexts(args.dataset, args.metadata)

    with args.dataset.open("w", encoding="utf-8") as handle:
        json.dump(updated_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[OK] {len(updated_rows)} lignes mises à jour dans {args.dataset}")


if __name__ == "__main__":
    main()

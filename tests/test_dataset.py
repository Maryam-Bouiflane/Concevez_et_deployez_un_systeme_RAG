"""Tests du chargement du dataset d'évaluation."""

from pathlib import Path

import pytest

from app.evaluation.dataset import (
    DATASET_PATH,
    load_evaluation_rows,
)


def test_load_evaluation_rows():
    """Le dataset est chargé sous forme de liste de dictionnaires."""

    rows = load_evaluation_rows()

    assert isinstance(rows, list)
    assert len(rows) > 0

    for row in rows:
        assert isinstance(row, dict)
        assert "user_input" in row
        assert "reference" in row


def test_dataset_path_exists():
    """Le fichier du dataset d'évaluation existe."""

    assert DATASET_PATH.exists()
    assert DATASET_PATH.is_file()


def test_evaluation_rows_have_valid_content():
    """Chaque exemple contient une question et une référence valides."""

    rows = load_evaluation_rows()

    for row in rows:
        assert isinstance(row["user_input"], str)
        assert row["user_input"].strip() != ""

        assert isinstance(row["reference"], str)
        assert row["reference"].strip() != ""


def test_default_city_placeholder_is_replaced():
    """Le placeholder de ville est remplacé par DEFAULT_CITY."""

    rows = load_evaluation_rows()

    for row in rows:
        for value in row.values():
            if isinstance(value, str):
                assert "{DEFAULT_CITY}" not in value


def test_load_evaluation_rows_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Une erreur est levée si le dataset n'existe pas."""

    missing_file = tmp_path / "missing.json"

    monkeypatch.setattr(
        "app.evaluation.dataset.DATASET_PATH",
        missing_file,
    )

    with pytest.raises(FileNotFoundError):
        load_evaluation_rows()


def test_load_evaluation_rows_invalid_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Une erreur est levée si le JSON n'est pas une liste."""

    invalid_file = tmp_path / "invalid.json"

    invalid_file.write_text(
        '{"user_input": "question"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.evaluation.dataset.DATASET_PATH",
        invalid_file,
    )

    with pytest.raises(ValueError, match="liste d'objets JSON"):
        load_evaluation_rows()

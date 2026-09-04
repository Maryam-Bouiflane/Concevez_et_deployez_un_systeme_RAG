"""Tests unitaires pour le chargement des événements Open Agenda."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from app.core.data_loader import (
    add_label,
    fetch_events_from_openagenda,
)


def test_add_label_adds_label_to_non_empty_values():
    series = pd.Series(["Paris", "", None, "Lyon"])

    result = add_label(series, "Ville: ")

    assert result.tolist() == [
        "Ville: Paris",
        "",
        "",
        "Ville: Lyon",
    ]


@patch("app.core.data_loader.requests.get")
def test_fetch_events_returns_normalized_events(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [
            {
                "uid": "event-1",
                "title_fr": "Concert",
                "slug": "concert",
                "description_fr": "Un concert",
                "longdescription_fr": "",
                "keywords_fr": "musique",
                "firstdate_begin": "2026-08-10",
                "lastdate_end": "2026-08-10",
                "location_name": "Salle",
                "location_address": "1 rue Test",
                "location_city": "Paris",
                "location_district": "",
                "location_postalcode": "75001",
                "location_department": "Paris",
                "location_region": "Île-de-France",
                "country_fr": "France",
                "location_coordinates.lon": 2.35,
                "location_coordinates.lat": 48.85,
            }
        ]
    }

    mock_get.return_value = mock_response

    result = fetch_events_from_openagenda(
        city="Paris",
        lookback_days=30,
    )

    assert len(result) == 1
    assert result[0]["uid"] == "event-1"
    assert result[0]["title_fr"] == "Concert"
    assert "Titre: Concert" in result[0]["text"]
    assert "Ville: Paris" in result[0]["text"]

    mock_get.assert_called_once()


@patch("app.core.data_loader.requests.get")
def test_fetch_events_returns_empty_list_when_no_events(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    result = fetch_events_from_openagenda()

    assert result == []


@patch("app.core.data_loader.requests.get")
def test_fetch_events_removes_duplicate_uids(mock_get):
    event = {
        "uid": "event-1",
        "title_fr": "Concert",
        "slug": "concert",
        "description_fr": "Description",
        "longdescription_fr": "",
        "keywords_fr": "",
        "firstdate_begin": "2026-08-10",
        "lastdate_end": "2026-08-10",
        "location_name": "Salle",
        "location_address": "",
        "location_city": "Paris",
        "location_district": "",
        "location_postalcode": "",
        "location_department": "",
        "location_region": "",
        "country_fr": "France",
        "location_coordinates.lon": 2.35,
        "location_coordinates.lat": 48.85,
    }

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [event, event]
    }

    mock_get.return_value = mock_response

    result = fetch_events_from_openagenda()

    assert len(result) == 1
    assert result[0]["uid"] == "event-1"


@patch("app.core.data_loader.requests.get")
def test_fetch_events_wraps_connection_error(mock_get):
    mock_get.side_effect = requests.RequestException(
        "Connection failed"
    )

    with pytest.raises(RuntimeError, match="Erreur de connexion"):
        fetch_events_from_openagenda()
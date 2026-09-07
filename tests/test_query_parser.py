"""Tests unitaires du Query Parser."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

import pytest

from app.core.query_parser import EventQueryParser
from app.schemas.search import EventSearchFilters, EventSearchQuery
from langchain_core.runnables import RunnableLambda

@pytest.fixture
def structured_llm() -> Mock:
    """Mock du LLM structuré."""
    return Mock()


@pytest.fixture
def mock_llm(
    structured_llm: Mock,
) -> Mock:
    """
    Mock du LLM utilisé par EventQueryParser.

    Le Runnable conserve l'entrée reçue afin que les tests
    puissent vérifier que la question originale est transmise.
    """

    llm = Mock()

    runnable = RunnableLambda(
        lambda value: structured_llm.invoke(value)
    )

    llm.with_structured_output.return_value = runnable

    return llm

@pytest.fixture
def parser(mock_llm: Mock) -> EventQueryParser:
    """Crée un EventQueryParser avec un LLM mocké."""
    return EventQueryParser(llm=mock_llm)


def test_parse_weekend_and_city(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le parser extrait correctement un week-end et une ville."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 5),
            date_to=date(2026, 9, 6),
            location_city="Paris",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu ce week-end à Paris ?"
    )

    assert result.filters is not None

    assert result.filters.date_from == date(2026, 9, 5)
    assert result.filters.date_to == date(2026, 9, 6)
    assert result.filters.location_city == "Paris"


def test_parse_tomorrow_and_city(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le parser extrait correctement demain et la ville."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 5),
            date_to=date(2026, 9, 5),
            location_city="Lyon",
        )
    )

    result = parser.parse(
        "Quels événements auront lieu demain à Lyon ?"
    )

    assert result.filters is not None

    assert result.filters.date_from == date(2026, 9, 5)
    assert result.filters.date_to == date(2026, 9, 5)
    assert result.filters.location_city == "Lyon"


def test_parse_relative_future_date(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le parser extrait une date relative."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 6),
            date_to=date(2026, 9, 6),
            location_city="Marseille",
        )
    )

    result = parser.parse(
        "Quels concerts sont prévus dans 2 jours à Marseille ?"
    )

    assert result.filters is not None

    assert result.filters.date_from == date(2026, 9, 6)
    assert result.filters.date_to == date(2026, 9, 6)
    assert result.filters.location_city == "Marseille"


def test_parse_next_week(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le parser extrait correctement la semaine prochaine."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
            location_city="Paris",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu la semaine prochaine à Paris ?"
    )

    assert result.filters is not None

    assert result.filters.date_from == date(2026, 9, 7)
    assert result.filters.date_to == date(2026, 9, 13)
    assert result.filters.location_city == "Paris"


def test_parse_month(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le parser extrait correctement un mois."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 31),
            location_city="Paris",
        )
    )

    result = parser.parse(
        "Quels événements sont prévus en août à Paris ?"
    )

    assert result.filters is not None

    assert result.filters.date_from == date(2026, 8, 1)
    assert result.filters.date_to == date(2026, 8, 31)
    assert result.filters.location_city == "Paris"


def test_parse_semantic_only_question_returns_no_filters(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Une question purement sémantique ne produit aucun filtre."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=None
    )

    result = parser.parse(
        "Quels sont les meilleurs concerts de jazz ?"
    )

    assert result.filters is None


def test_parse_explicit_country(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le pays explicitement indiqué est extrait."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            country_fr="France",
        )
    )

    result = parser.parse(
        "Quels événements sont organisés en France ?"
    )

    assert result.filters is not None
    assert result.filters.country_fr == "France"


def test_parse_explicit_region(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """La région explicitement indiquée est extraite."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_region="Île-de-France",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu en Île-de-France ?"
    )

    assert result.filters is not None
    assert result.filters.location_region == "Île-de-France"


def test_parse_explicit_department(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le département explicitement indiqué est extrait."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_department="Rhône",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu dans le Rhône ?"
    )

    assert result.filters is not None
    assert result.filters.location_department == "Rhône"


def test_parse_explicit_district(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le quartier ou arrondissement explicitement indiqué est extrait."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_district="1er arrondissement",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu dans le 1er arrondissement ?"
    )

    assert result.filters is not None
    assert (
        result.filters.location_district
        == "1er arrondissement"
    )


def test_parse_explicit_postal_code(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Le code postal explicitement indiqué est extrait."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_postalcode="75001",
        )
    )

    result = parser.parse(
        "Quels événements ont lieu dans le 75001 ?"
    )

    assert result.filters is not None
    assert result.filters.location_postalcode == "75001"


def test_parse_does_not_infer_geography(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """
    Le parser ne doit pas inventer de filtre géographique.

    La question ne contient aucune ville, région ou autre
    information géographique explicite.
    """

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=None
    )

    result = parser.parse(
        "Quels événements de musique sont intéressants ?"
    )

    assert result.filters is None


def test_parse_explicit_date(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Une date explicitement indiquée est extraite."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 15),
            date_to=date(2026, 9, 15),
        )
    )

    result = parser.parse(
        "Quels événements sont prévus le 15 septembre 2026 ?"
    )

    assert result.filters is not None
    assert result.filters.date_from == date(2026, 9, 15)
    assert result.filters.date_to == date(2026, 9, 15)


def test_parse_passes_original_question_to_llm(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """Vérifie que la question complète est utilisée."""

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        )
    )

    question = (
        "Quels concerts de jazz sont prévus ce week-end "
        "à Paris ?"
    )

    parser.parse(question)

    structured_llm.invoke.assert_called_once()

    prompt_value = structured_llm.invoke.call_args.args[0]

    assert question in str(prompt_value)


def test_parse_converts_empty_filters_to_none(
    parser: EventQueryParser,
    structured_llm: Mock,
) -> None:
    """
    Si le LLM retourne un objet EventSearchFilters vide,
    le parser doit produire filters=None.
    """

    structured_llm.invoke.return_value = EventSearchQuery(
        filters=EventSearchFilters()
    )

    result = parser.parse(
        "Quels sont les événements intéressants ?"
    )

    assert result.filters is None
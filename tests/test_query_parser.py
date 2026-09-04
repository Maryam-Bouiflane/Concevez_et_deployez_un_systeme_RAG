from langchain_openai import ChatOpenAI

from app.config import MISTRAL_API_KEY
from app.core.query_parser import EventQueryParser


MISTRAL_LLM_MODEL = "mistral-small-latest"


llm = ChatOpenAI(
    model=MISTRAL_LLM_MODEL,
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
    temperature=0.0,
    max_tokens=1024,
)


parser = EventQueryParser(
    llm=llm,
)


def test_weekend_with_city() -> None:
    """Vérifie l'extraction du week-end et de la ville."""

    question = "Quels événements ont lieu ce week-end à Paris ?"

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.date_from is not None
    assert result.filters.date_to is not None

    assert result.filters.location_city == "Paris"

    # Le parser ne doit pas déduire ces informations
    # à partir de Paris.
    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_tomorrow_with_city() -> None:
    """Vérifie l'extraction de 'demain' et de la ville."""

    question = "Quels événements auront lieu demain à Lyon ?"

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.date_from is not None
    assert result.filters.date_to is not None

    assert (
        result.filters.date_from
        == result.filters.date_to
    )

    assert result.filters.location_city == "Lyon"

    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_future_date_with_city() -> None:
    """Vérifie l'extraction d'une date relative."""

    question = (
        "Quels concerts sont prévus dans 2 jours "
        "à Marseille ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.date_from is not None
    assert result.filters.date_to is not None

    assert (
        result.filters.date_from
        == result.filters.date_to
    )

    assert result.filters.location_city == "Marseille"

    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_next_week_with_city() -> None:
    """Vérifie l'extraction de la semaine prochaine."""

    question = (
        "Quels événements ont lieu la semaine prochaine "
        "à Paris ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.date_from is not None
    assert result.filters.date_to is not None

    assert (
        result.filters.date_from
        < result.filters.date_to
    )

    assert result.filters.location_city == "Paris"

    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_month_with_city() -> None:
    """Vérifie l'extraction d'un mois explicite."""

    question = "Quels événements sont prévus en août à Paris ?"

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.date_from is not None
    assert result.filters.date_to is not None

    assert result.filters.date_from.month == 8
    assert result.filters.date_to.month == 8

    assert result.filters.location_city == "Paris"

    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_no_metadata_filter() -> None:
    """
    Vérifie qu'une question purement sémantique
    retourne filters=None.
    """

    question = "Quels concerts de jazz sont proposés ?"

    result = parser.parse(question)

    assert result.filters is None


def test_explicit_country() -> None:
    """Vérifie qu'un pays explicitement indiqué est extrait."""

    question = "Quels événements sont organisés en France ?"

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.country_fr == "France"

    # Rien ne doit être déduit automatiquement.
    assert result.filters.location_city is None
    assert result.filters.location_region is None
    assert result.filters.location_department is None
    assert result.filters.location_countrycode is None


def test_explicit_region() -> None:
    """Vérifie qu'une région explicitement indiquée est extraite."""

    question = (
        "Quels événements sont organisés "
        "en Île-de-France ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.location_region == "Île-de-France"

    assert result.filters.location_city is None
    assert result.filters.location_department is None
    assert result.filters.country_fr is None


def test_explicit_department() -> None:
    """Vérifie qu'un département explicitement indiqué est extrait."""

    question = (
        "Quels événements sont organisés "
        "dans le département 75 ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.location_department == "75"

    assert result.filters.location_city is None
    assert result.filters.location_region is None
    assert result.filters.country_fr is None


def test_explicit_district() -> None:
    """Vérifie qu'un arrondissement explicitement indiqué est extrait."""

    question = (
        "Quels événements sont organisés "
        "à Paris dans le 15e arrondissement ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.location_city == "Paris"
    assert result.filters.location_district == "15e"

    # Aucun autre champ géographique ne doit être déduit.
    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.location_countrycode is None
    assert result.filters.country_fr is None


def test_explicit_postal_code() -> None:
    """Vérifie qu'un code postal explicitement indiqué est extrait."""

    question = (
        "Quels événements sont organisés "
        "dans le 75015 ?"
    )

    result = parser.parse(question)

    assert result.filters is not None

    assert result.filters.location_postalcode == "75015"

    assert result.filters.location_city is None
    assert result.filters.location_department is None
    assert result.filters.location_region is None
    assert result.filters.country_fr is None

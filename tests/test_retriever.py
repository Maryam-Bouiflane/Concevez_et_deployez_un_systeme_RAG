"""Tests unitaires du retriever."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from app.core.retriever import EventRetriever
from app.schemas.search import EventSearchFilters, EventSearchQuery


class FakeIndex:
    """Faux index permettant de tester EventRetriever sans Mistral."""

    def __init__(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> None:
        self.documents = documents
        self.embeddings = embeddings.astype("float32")

        self.index = Mock()
        self.index.ntotal = len(documents)

        self.create_query_embedding = Mock(
            return_value=np.array(
                [[1.0, 0.0]],
                dtype="float32",
            )
        )

    def get_embedding(
        self,
        index: int,
    ) -> np.ndarray:
        """Retourne le vecteur correspondant à un document."""

        return self.embeddings[index]


def make_document(
    uid: str,
    title: str,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    city: str | None = None,
) -> Document:
    """Crée un Document de test."""

    metadata = {
        "uid": uid,
        "title": title,
        "date_start": date_start,
        "date_end": date_end,
        "location_city": city,
        "location_district": None,
        "location_postalcode": None,
        "location_department": None,
        "location_region": None,
        "location_countrycode": None,
        "country": None,
    }

    return Document(
        id=uid,
        page_content=title,
        metadata=metadata,
    )


@pytest.fixture
def documents() -> list[Document]:
    """Documents utilisés dans les tests."""

    return [
        make_document(
            "1",
            "Concert de jazz à Paris",
            date_start="2026-08-15T20:00:00",
            date_end="2026-08-15T22:00:00",
            city="Paris",
        ),
        make_document(
            "2",
            "Exposition à Lyon",
            date_start="2026-08-15T10:00:00",
            date_end="2026-08-20T18:00:00",
            city="Lyon",
        ),
        make_document(
            "3",
            "Festival à Paris",
            date_start="2026-08-20T10:00:00",
            date_end="2026-08-22T22:00:00",
            city="Paris",
        ),
        make_document(
            "4",
            "Théâtre à Marseille",
            date_start="2026-09-01T20:00:00",
            date_end="2026-09-01T22:00:00",
            city="Marseille",
        ),
    ]


@pytest.fixture
def fake_index(
    documents: list[Document],
) -> FakeIndex:
    """Crée un faux index."""

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype="float32",
    )

    return FakeIndex(
        documents,
        embeddings,
    )


@pytest.fixture
def query_parser() -> Mock:
    """Mock du Query Parser pour éviter les appels Mistral."""

    parser = Mock()

    parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    return parser


@pytest.fixture
def retriever(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> EventRetriever:
    """Crée un EventRetriever de test."""

    return EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )


def configure_faiss_search(
    fake_index: FakeIndex,
    scores: list[float],
    indices: list[int],
) -> None:
    """Configure un résultat FAISS simulé."""

    def search(
        query_embedding: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del query_embedding

        return (
            np.array(
                [scores[:k]],
                dtype="float32",
            ),
            np.array(
                [indices[:k]],
                dtype="int64",
            ),
        )

    fake_index.index.search.side_effect = search


# ======================================================================
# SEARCH PUBLIC
# ======================================================================


def test_search_returns_documents_with_scores(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """search() retourne les Documents avec leurs scores."""

    question = "Quels événements sont intéressants ?"

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95, 0.80, 0.30],
        indices=[0, 1, 2],
    )

    results = retriever.search(question)

    assert len(results) == 2

    assert results[0]["score"] == pytest.approx(0.95)
    assert results[1]["score"] == pytest.approx(0.80)

    assert results[0]["document"].metadata["uid"] == "1"
    assert results[1]["document"].metadata["uid"] == "2"


def test_search_keeps_scores_outside_document_metadata(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Le score ne doit jamais être ajouté aux metadata."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95],
        indices=[0],
    )

    results = retriever.search(
        "Quels événements sont intéressants ?"
    )

    document = results[0]["document"]

    assert results[0]["score"] == pytest.approx(0.95)
    assert "score" not in document.metadata


def test_search_stores_last_parsed_query(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """search() conserve le résultat du Query Parser."""

    parsed_query = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
            date_from=date(2026, 8, 15),
            date_to=date(2026, 8, 16),
        ),
    )

    query_parser.parse.return_value = parsed_query

    configure_faiss_search(
        fake_index,
        scores=[0.95],
        indices=[0],
    )

    retriever.search(
        "Quels événements sont prévus demain à Paris ?"
    )

    assert retriever.last_parsed_query is parsed_query


def test_search_calls_query_parser_once(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Une recherche appelle le Query Parser exactement une fois."""

    question = "Quels événements sont prévus à Paris ?"

    configure_faiss_search(
        fake_index,
        scores=[0.95],
        indices=[0],
    )

    retriever.search(question)

    query_parser.parse.assert_called_once_with(question)


def test_search_rejects_empty_query(
    retriever: EventRetriever,
) -> None:
    """Une requête vide est refusée."""

    with pytest.raises(
        ValueError,
        match="requête de recherche",
    ):
        retriever.search("")


def test_search_rejects_whitespace_query(
    retriever: EventRetriever,
) -> None:
    """Une requête composée uniquement d'espaces est refusée."""

    with pytest.raises(
        ValueError,
        match="requête de recherche",
    ):
        retriever.search("   ")


# ======================================================================
# LANGCHAIN RETRIEVER
# ======================================================================


def test_get_relevant_documents_returns_documents_without_scores(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """
    _get_relevant_documents() retourne uniquement les Documents.

    Les scores restent disponibles via search().
    """

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95, 0.80, 0.30],
        indices=[0, 1, 2],
    )

    results = retriever._get_relevant_documents(
        "Quels événements sont intéressants ?",
        run_manager=Mock(),
    )

    assert len(results) == 2

    assert all(
        isinstance(document, Document)
        for document in results
    )

    assert results[0].metadata["uid"] == "1"
    assert results[1].metadata["uid"] == "2"

    assert "score" not in results[0].metadata
    assert "score" not in results[1].metadata


def test_get_relevant_documents_uses_public_search(
    retriever: EventRetriever,
) -> None:
    """
    _get_relevant_documents() doit utiliser search().

    Les scores retournés par search() sont retirés avant de
    retourner les Documents à LangChain.
    """

    document = Document(
        id="1",
        page_content="Concert",
        metadata={"uid": "1"},
    )

    mock_search = Mock(
        return_value=[
            {
                "score": 0.91,
                "document": document,
            }
        ]
    )

    with patch.object(
        EventRetriever,
        "search",
        mock_search,
    ):
        results = retriever._get_relevant_documents(
            "Concert",
            run_manager=Mock(),
        )

    mock_search.assert_called_once_with("Concert")

    assert results == [document]


# ======================================================================
# METADATA FILTERING
# ======================================================================


def test_metadata_filter_by_city(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Un filtre de ville ne conserve que les événements concernés."""

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        location_city="Paris",
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert candidate_indices == [0, 2]


def test_metadata_filter_is_case_insensitive(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les filtres géographiques sont insensibles à la casse."""

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        location_city="paris",
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert candidate_indices == [0, 2]


def test_metadata_filter_by_date(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les événements qui chevauchent la période sont conservés."""

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        date_from=date(2026, 8, 15),
        date_to=date(2026, 8, 16),
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert candidate_indices == [0, 1]


def test_metadata_filter_by_date_excludes_events_after_period(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les événements entièrement après la période sont exclus."""

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        date_from=date(2026, 8, 15),
        date_to=date(2026, 8, 16),
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert 2 not in candidate_indices
    assert 3 not in candidate_indices


def test_metadata_filters_city_and_date(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les filtres metadata sont combinés avec AND."""

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        location_city="Paris",
        date_from=date(2026, 8, 20),
        date_to=date(2026, 8, 22),
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert candidate_indices == [2]


def test_metadata_filter_by_country(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Le filtre country utilise metadata['country'].""" 

    fake_index.documents[0].metadata["country"] = "France"
    fake_index.documents[1].metadata["country"] = "France"
    fake_index.documents[2].metadata["country"] = "Belgique"

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    filters = EventSearchFilters(
        country_fr="France",
    )

    candidate_indices = (
        retriever._get_matching_document_indices(filters)
    )

    assert candidate_indices == [0, 1]


# ======================================================================
# FILTERED SEARCH
# ======================================================================


def test_filtered_search_uses_only_matching_documents(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """La recherche filtrée utilise uniquement les candidats metadata."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        ),
    )

    results = retriever.search(
        "Quels concerts sont prévus à Paris ?"
    )

    assert len(results) == 2

    assert results[0]["document"].metadata["uid"] == "1"
    assert results[1]["document"].metadata["uid"] == "3"

    assert "score" in results[0]
    assert "score" in results[1]

    fake_index.index.search.assert_not_called()


def test_filtered_search_preserves_scores(
    retriever: EventRetriever,
    query_parser: Mock,
) -> None:
    """Une recherche filtrée retourne les scores FAISS."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        ),
    )

    results = retriever.search(
        "Quels événements sont prévus à Paris ?"
    )

    assert len(results) == 2

    # Query embedding = [1, 0]
    #
    # Document 1 = [1, 0] -> score = 1.0
    # Document 3 = [0.8, 0.2] -> score = 0.8

    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.8)


def test_filtered_search_does_not_use_global_faiss(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Avec un filtre, le FAISS principal ne reçoit aucun search()."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        ),
    )

    retriever.search(
        "Quels événements sont prévus à Paris ?"
    )

    fake_index.index.search.assert_not_called()


def test_filtered_search_respects_top_k(
    retriever: EventRetriever,
    query_parser: Mock,
) -> None:
    """Une recherche filtrée respecte également top_k."""

    retriever.top_k = 1

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        ),
    )

    results = retriever.search(
        "Quels événements sont prévus à Paris ?"
    )

    assert len(results) == 1
    assert results[0]["document"].metadata["uid"] == "1"


# ======================================================================
# THRESHOLD
# ======================================================================


def test_threshold_removes_low_scores(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les résultats sous le threshold sont supprimés."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.90, 0.70, 0.40],
        indices=[0, 1, 2],
    )

    results = retriever.search(
        "Quels événements sont proposés ?"
    )

    assert len(results) == 2

    assert results[0]["document"].metadata["uid"] == "1"
    assert results[1]["document"].metadata["uid"] == "2"

    assert results[0]["score"] == pytest.approx(0.90)
    assert results[1]["score"] == pytest.approx(0.70)


def test_threshold_is_inclusive(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Un score exactement égal au threshold est conservé."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    retriever.threshold = 0.80

    configure_faiss_search(
        fake_index,
        scores=[0.80],
        indices=[0],
    )

    results = retriever.search(
        "Quels événements sont proposés ?"
    )

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.80)


# ======================================================================
# TOP K
# ======================================================================


def test_top_k_limits_number_of_results(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """top_k limite le nombre de résultats."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95, 0.90, 0.85],
        indices=[0, 1, 2],
    )

    retriever.top_k = 2

    results = retriever.search(
        "Quels événements sont proposés ?"
    )

    assert len(results) == 2

    fake_index.index.search.assert_called_once()

    _, search_k = fake_index.index.search.call_args.args

    assert search_k == 2


def test_top_k_is_limited_by_index_size(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """search_k ne peut pas dépasser la taille de l'index."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    retriever.top_k = 100

    configure_faiss_search(
        fake_index,
        scores=[0.95, 0.90, 0.85, 0.70],
        indices=[0, 1, 2, 3],
    )

    results = retriever.search(
        "Quels événements sont proposés ?"
    )

    assert len(results) == 4

    fake_index.index.search.assert_called_once()

    _, search_k = fake_index.index.search.call_args.args

    assert search_k == 4


# ======================================================================
# EMPTY RESULTS / STRICT FILTERING
# ======================================================================


def test_no_matching_documents_returns_empty_list(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Aucun candidat metadata retourne une liste vide."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Bordeaux",
        ),
    )

    results = retriever.search(
        "Quels événements sont prévus à Bordeaux ?"
    )

    assert results == []

    assert fake_index.index.search.call_count == 0
    assert fake_index.create_query_embedding.call_count == 0


def test_explicit_filter_does_not_fallback_to_unfiltered_documents(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Un filtre sans correspondance ne provoque aucun fallback."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Bordeaux",
        ),
    )

    results = retriever.search(
        "Quels événements sont prévus à Bordeaux ?"
    )

    assert results == []

    assert fake_index.index.search.call_count == 0
    assert fake_index.create_query_embedding.call_count == 0


# ======================================================================
# QUESTION / EMBEDDING
# ======================================================================


def test_original_question_is_sent_to_embedding(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """
    Le retriever envoie la question originale complète au modèle
    d'embedding.
    """

    question = (
        "Quels concerts de jazz sont prévus ce week-end "
        "à Paris pour les familles ?"
    )

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95],
        indices=[0],
    )

    retriever.search(question)

    fake_index.create_query_embedding.assert_called_once_with(
        question
    )


# ======================================================================
# INDEX
# ======================================================================


def test_loads_index_when_faiss_index_is_missing(
    documents: list[Document],
    query_parser: Mock,
) -> None:
    """load() est appelé lorsque l'index FAISS est absent."""

    fake_index = FakeIndex(
        documents,
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype="float32",
        ),
    )

    fake_index.index = None
    fake_index.load = Mock()

    loaded_index = Mock()
    loaded_index.ntotal = 4

    def load() -> None:
        fake_index.index = loaded_index

    fake_index.load.side_effect = load

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    loaded_index.search.return_value = (
        np.array([[0.95]], dtype="float32"),
        np.array([[0]], dtype="int64"),
    )

    results = retriever.search(
        "Quels événements sont intéressants ?"
    )

    fake_index.load.assert_called_once()

    assert len(results) == 1
    assert results[0]["document"].metadata["uid"] == "1"


def test_raises_when_index_remains_unavailable(
    documents: list[Document],
    query_parser: Mock,
) -> None:
    """Une RuntimeError est levée si l'index reste indisponible."""

    fake_index = FakeIndex(
        documents,
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype="float32",
        ),
    )

    fake_index.index = None
    fake_index.load = Mock()

    retriever = EventRetriever(
        index=fake_index,
        query_parser=query_parser,
        top_k=3,
        threshold=0.5,
    )

    query_parser.parse.return_value = EventSearchQuery(
        filters=None,
    )

    with pytest.raises(
        RuntimeError,
        match="index FAISS",
    ):
        retriever.search(
            "Quels événements sont intéressants ?"
        )

    fake_index.load.assert_called_once()

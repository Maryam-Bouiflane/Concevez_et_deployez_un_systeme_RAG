"""Tests unitaires du retriever."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

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

    def get_embedding(self, index: int) -> np.ndarray:
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

    return Mock()


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
    """
    Configure un résultat FAISS simulé.

    Le mock respecte le paramètre k reçu par FAISS.
    Cela permet notamment de tester correctement top_k.
    """

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


def test_semantic_search_without_filters(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Sans filtres, FAISS recherche dans tout l'index."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None
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

    assert results[0].metadata["uid"] == "1"
    assert results[1].metadata["uid"] == "2"

    query_parser.parse.assert_called_once_with(
        "Quels événements sont intéressants ?"
    )

    fake_index.index.search.assert_called_once()


def test_metadata_filter_by_city(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Un filtre de ville ne conserve que les événements de cette ville."""

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
        retriever._get_matching_document_indices(
            filters
        )
    )

    assert candidate_indices == [0, 2]


def test_metadata_filter_by_date(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les événements qui chevauchent la période recherchée sont conservés."""

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
        retriever._get_matching_document_indices(
            filters
        )
    )

    assert candidate_indices == [0, 1]


def test_metadata_filters_city_and_date(
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Plusieurs filtres metadata sont combinés avec AND."""

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
        retriever._get_matching_document_indices(
            filters
        )
    )

    assert candidate_indices == [2]


def test_filtered_search_uses_only_matching_documents(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """
    Vérifie le pipeline :

        parser
        ↓
        metadata filter
        ↓
        FAISS temporaire sur les candidats
    """

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Paris",
        )
    )

    # Les deux documents correspondant à Paris
    # sont les documents globaux 0 et 2.
    #
    # Le FAISS temporaire travaille donc avec
    # les deux vecteurs correspondants.
    results = retriever._get_relevant_documents(
        "Quels concerts sont prévus à Paris ?",
        run_manager=Mock(),
    )

    assert len(results) == 2

    assert results[0].metadata["uid"] == "1"
    assert results[1].metadata["uid"] == "3"

    # Le FAISS global ne doit pas être utilisé
    # lorsqu'un filtre metadata est présent.
    fake_index.index.search.assert_not_called()


def test_threshold_removes_low_scores(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Les résultats sous le threshold sont supprimés."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None
    )

    configure_faiss_search(
        fake_index,
        scores=[0.90, 0.70, 0.40],
        indices=[0, 1, 2],
    )

    results = retriever._get_relevant_documents(
        "Quels événements sont proposés ?",
        run_manager=Mock(),
    )

    assert len(results) == 2

    assert results[0].metadata["uid"] == "1"
    assert results[1].metadata["uid"] == "2"


def test_top_k_limits_number_of_results(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """top_k limite le nombre de résultats."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=None
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95, 0.90, 0.85],
        indices=[0, 1, 2],
    )

    retriever.top_k = 2

    results = retriever._get_relevant_documents(
        "Quels événements sont proposés ?",
        run_manager=Mock(),
    )

    assert len(results) == 2

    # Vérifie également que top_k=2 a bien été transmis à FAISS.
    fake_index.index.search.assert_called_once()

    _, search_k = fake_index.index.search.call_args.args

    assert search_k == 2


def test_no_matching_documents_returns_empty_list(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """Aucun candidat metadata doit retourner une liste vide."""

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Bordeaux",
        )
    )

    results = retriever._get_relevant_documents(
        "Quels événements sont prévus à Bordeaux ?",
        run_manager=Mock(),
    )

    assert results == []

    fake_index.index.search.assert_not_called()


def test_explicit_filter_does_not_fallback_to_unfiltered_documents(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """
    Un filtre explicite qui ne correspond à aucun document
    ne doit jamais provoquer un fallback vers tout l'index.
    """

    query_parser.parse.return_value = EventSearchQuery(
        filters=EventSearchFilters(
            location_city="Bordeaux",
        )
    )

    results = retriever._get_relevant_documents(
        "Quels événements sont prévus à Bordeaux ?",
        run_manager=Mock(),
    )

    assert results == []

    fake_index.index.search.assert_not_called()


def test_original_question_is_sent_to_embedding(
    retriever: EventRetriever,
    fake_index: FakeIndex,
    query_parser: Mock,
) -> None:
    """
    Le retriever doit envoyer la question originale complète
    au modèle d'embedding.

    Le Query Parser extrait les filtres mais ne réécrit pas
    la requête sémantique.
    """

    question = (
        "Quels concerts de jazz sont prévus ce week-end "
        "à Paris pour les familles ?"
    )

    query_parser.parse.return_value = EventSearchQuery(
        filters=None
    )

    configure_faiss_search(
        fake_index,
        scores=[0.95],
        indices=[0],
    )

    retriever._get_relevant_documents(
        question,
        run_manager=Mock(),
    )

    fake_index.create_query_embedding.assert_called_once_with(
        question
    )
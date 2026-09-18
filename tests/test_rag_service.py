"""Tests unitaires du service RAG."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.core.rag_service import EventRAGService
from app.schemas.search import EventSearchFilters, EventSearchQuery


@pytest.fixture
def service() -> EventRAGService:
    """
    Crée un EventRAGService minimal pour les tests.

    On ne construit ni Mistral, ni Query Parser, ni vraie chaîne
    LangChain : ces composants sont mockés.

    Le Retriever est responsable du Query Parser dans la nouvelle
    architecture. Le service récupère uniquement le résultat déjà
    parsé via ``retriever.last_parsed_query``.
    """

    service = EventRAGService.__new__(EventRAGService)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    service.index = Mock()

    # L'index est considéré comme déjà chargé.
    service.index.index = Mock()

    service.index.documents = [
        Document(
            id="event-1",
            page_content=(
                "Concert de jazz à Paris "
                "le 15 août 2026."
            ),
            metadata={
                "uid": "event-1",
                "title": "Concert de jazz à Paris",
            },
        ),
        Document(
            id="event-2",
            page_content="Festival de jazz à Paris.",
            metadata={
                "uid": "event-2",
                "title": "Festival de jazz à Paris",
            },
        ),
    ]

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------

    service.retriever = Mock()

    # Le Retriever est maintenant responsable du parsing.
    # Le service récupère simplement le résultat déjà produit.
    service.retriever.last_parsed_query = EventSearchQuery(
        filters=None
    )

    # Par défaut, le Retriever retourne les deux documents avec
    # leurs scores FAISS.
    service.retriever.search.return_value = [
        {
            "score": 0.94,
            "document": service.index.documents[0],
        },
        {
            "score": 0.81,
            "document": service.index.documents[1],
        },
    ]

    # ------------------------------------------------------------------
    # Query Parser
    # ------------------------------------------------------------------

    # Le service n'utilise plus directement le Query Parser dans answer().
    # On le conserve néanmoins pour représenter la structure complète
    # du service dans les tests.
    service.query_parser = Mock()

    # ------------------------------------------------------------------
    # Chaîne LangChain
    # ------------------------------------------------------------------

    service._document_chain = Mock()

    service._document_chain.invoke.return_value = (
        "Le concert de jazz a lieu à Paris "
        "le 15 août 2026."
    )

    return service


def test_answer_returns_generated_answer(
    service: EventRAGService,
) -> None:
    """Le service retourne correctement la réponse générée."""

    question = "Quels concerts sont prévus à Paris ?"

    result = service.answer(question)

    assert result["answer"] == (
        "Le concert de jazz a lieu à Paris "
        "le 15 août 2026."
    )

    # Les scores ne doivent pas être transmis au LLM.
    service._document_chain.invoke.assert_called_once_with(
        {
            "input": question,
            "context": service.index.documents,
        }
    )


def test_answer_returns_context_documents(
    service: EventRAGService,
) -> None:
    """
    Le service retourne les documents avec leurs scores
    dans le contexte destiné à l'API.
    """

    question = "Quels concerts sont prévus à Paris ?"

    result = service.answer(question)

    assert "context" in result
    assert len(result["context"]) == 2

    assert (
        result["context"][0]["document"].metadata["uid"]
        == "event-1"
    )

    assert (
        result["context"][1]["document"].metadata["uid"]
        == "event-2"
    )


def test_answer_keeps_retriever_scores_in_context(
    service: EventRAGService,
) -> None:
    """
    Le service conserve les scores FAISS pour l'API.

    Les scores restent dans ``result["context"]`` mais ne sont
    pas transmis au LLM.
    """

    service.retriever.search.return_value = [
        {
            "score": 0.94,
            "document": service.index.documents[0],
        },
        {
            "score": 0.81,
            "document": service.index.documents[1],
        },
    ]

    result = service.answer(
        "Quels événements sont prévus à Paris ?"
    )

    assert len(result["context"]) == 2

    assert result["context"][0]["score"] == 0.94
    assert result["context"][1]["score"] == 0.81

    assert (
        result["context"][0]["document"]
        == service.index.documents[0]
    )

    assert (
        result["context"][1]["document"]
        == service.index.documents[1]
    )

    # Vérifie que les scores ne sont PAS transmis au LLM.
    service._document_chain.invoke.assert_called_once_with(
        {
            "input": "Quels événements sont prévus à Paris ?",
            "context": service.index.documents,
        }
    )


def test_answer_marks_mistral_as_used(
    service: EventRAGService,
) -> None:
    """Avec l'option 1, Mistral est toujours utilisé."""

    result = service.answer(
        "Quels événements sont prévus à Paris ?"
    )

    assert result["used_mistral"] is True


def test_answer_rejects_empty_question(
    service: EventRAGService,
) -> None:
    """Une question vide est refusée."""

    with pytest.raises(ValueError, match="question"):
        service.answer("")


def test_answer_rejects_whitespace_question(
    service: EventRAGService,
) -> None:
    """Une question contenant uniquement des espaces est refusée."""

    with pytest.raises(ValueError, match="question"):
        service.answer("   ")


def test_answer_uses_single_retrieval_pass(
    service: EventRAGService,
) -> None:
    """
    Le service effectue une seule recherche.

    Il ne relance pas le Query Parser ni le Retriever après
    la première recherche.
    """

    question = "Quels événements sont prévus à Paris ?"

    service.answer(question)

    assert service.retriever.search.call_count == 1
    service.retriever.search.assert_called_once_with(question)

    # Le Query Parser ne doit pas être appelé directement
    # par EventRAGService.answer().
    service.query_parser.parse.assert_not_called()


def test_answer_sends_documents_without_scores_to_llm(
    service: EventRAGService,
) -> None:
    """
    Les scores restent disponibles pour l'API mais ne sont jamais
    transmis au LLM.
    """

    service.retriever.search.return_value = [
        {
            "score": 0.94,
            "document": service.index.documents[0],
        },
        {
            "score": 0.81,
            "document": service.index.documents[1],
        },
    ]

    question = "Quels événements sont prévus à Paris ?"

    service.answer(question)

    service._document_chain.invoke.assert_called_once()

    invoke_arguments = (
        service._document_chain.invoke.call_args.args[0]
    )

    assert invoke_arguments["input"] == question
    assert invoke_arguments["context"] == service.index.documents

    # Aucun dictionnaire contenant un score ne doit être transmis
    # au contexte du LLM.
    assert all(
        not isinstance(document, dict)
        for document in invoke_arguments["context"]
    )


def test_answer_rejects_date_not_in_context(
    service: EventRAGService,
) -> None:
    """
    Une date générée par le LLM mais située en dehors de la période
    demandée est rejetée.
    """

    # Dans la nouvelle architecture, le Query Parser est exécuté
    # par le Retriever. Le service récupère donc le résultat via
    # ``retriever.last_parsed_query``.
    service.retriever.last_parsed_query = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 5),
            date_to=date(2026, 9, 5),
            location_city="Paris",
        )
    )

    service._document_chain.invoke.return_value = (
        "Voici les événements prévus demain à Paris "
        "(le 20 juin 2025) :"
    )

    result = service.answer(
        "Quels événements sont prévus demain à Paris ?"
    )

    assert "20 juin 2025" not in result["answer"]

    assert (
        "Je n'ai pas trouvé d'événement correspondant"
        in result["answer"]
    )


def test_answer_uses_last_parsed_query_from_retriever(
    service: EventRAGService,
) -> None:
    """
    La validation des dates utilise le Query Parser déjà exécuté
    par le Retriever et ne relance pas le parsing.
    """

    parsed_query = EventSearchQuery(
        filters=EventSearchFilters(
            date_from=date(2026, 9, 5),
            date_to=date(2026, 9, 5),
            location_city="Paris",
        )
    )

    service.retriever.last_parsed_query = parsed_query

    service._document_chain.invoke.return_value = (
        "Un événement est prévu le 5 septembre 2026."
    )

    result = service.answer(
        "Quels événements sont prévus demain à Paris ?"
    )

    assert result["answer"] == (
        "Un événement est prévu le 5 septembre 2026."
    )

    # Le Query Parser ne doit pas être rappelé par le service.
    service.query_parser.parse.assert_not_called()


def test_answer_returns_no_event_message_when_retrieval_is_empty(
    service: EventRAGService,
) -> None:
    """
    Si le Retriever ne retourne aucun document, le LLM n'est pas appelé
    et le service retourne directement le message prévu.
    """

    service.retriever.search.return_value = []

    result = service.answer(
        "Quels événements sont prévus à Paris ?"
    )

    assert result["answer"] == (
        "Je n'ai pas trouvé d'événement correspondant "
        "dans les informations disponibles."
    )

    assert result["context"] == []
    assert result["used_mistral"] is True

    service._document_chain.invoke.assert_not_called()
"""Tests unitaires du service RAG."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.core.rag_service import EventRAGService


@pytest.fixture
def service() -> EventRAGService:
    """
    Crée un EventRAGService minimal pour les tests.

    On ne construit ni Mistral, ni Query Parser, ni vraie chaîne
    LangChain : ces composants sont testés séparément.
    """

    service = EventRAGService.__new__(EventRAGService)

    service.index = Mock()
    service.retriever = Mock()

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

    service.rag_chain = Mock()

    service.rag_chain.invoke.return_value = {
        "answer": (
            "Le concert de jazz a lieu à Paris "
            "le 15 août 2026."
        ),
        "context": service.index.documents,
    }

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

    service.rag_chain.invoke.assert_called_once_with(
        {
            "input": question,
        }
    )


def test_answer_returns_context_documents(
    service: EventRAGService,
) -> None:
    """Le service retourne les documents utilisés comme contexte."""

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


def test_answer_does_not_call_fallback(
    service: EventRAGService,
) -> None:
    """
    Vérifie que le service passe directement par rag_chain.

    Il n'existe plus de branche de fallback.
    """

    question = "Quels événements sont prévus à Paris ?"

    service.answer(question)

    service.rag_chain.invoke.assert_called_once_with(
        {
            "input": question,
        }
    )


def test_answer_returns_none_score_when_using_langchain_context(
    service: EventRAGService,
) -> None:
    """
    Les Documents LangChain ne contiennent pas encore le score FAISS.
    Le service retourne donc score=None.
    """

    result = service.answer(
        "Quels événements sont prévus à Paris ?"
    )

    assert len(result["context"]) == 2

    for item in result["context"]:
        assert item["score"] is None
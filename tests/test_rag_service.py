"""Tests unitaires du service RAG."""

from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.core.rag_service import EventRAGService


def make_result(score=0.9, content="Concert à Paris"):
    return {
        "score": score,
        "document": Document(
            page_content=content,
            metadata={"uid": "event-1"},
        ),
    }


@patch("app.core.rag_service.EventRAGIndex")
def test_answer_returns_message_when_no_results(mock_index_class):
    mock_index = mock_index_class.return_value
    mock_index.index = Mock()
    mock_index.search.return_value = []

    service = EventRAGService()
    service.index = mock_index

    result = service.answer("Quel concert ?")

    assert result["used_mistral"] is False
    assert result["context"] == []
    assert "aucun événement" in result["answer"].lower()


@patch("app.core.rag_service.EventRAGIndex")
def test_answer_returns_context_without_mistral(
    mock_index_class,
):
    mock_index = mock_index_class.return_value
    mock_index.index = Mock()
    mock_index.search.return_value = [
        make_result()
    ]

    service = EventRAGService()
    service.index = mock_index
    service.client = None

    result = service.answer("Quel concert ?")

    assert result["used_mistral"] is False
    assert len(result["context"]) == 1
    assert "Concert à Paris" in result["answer"]


@patch("app.core.rag_service.EventRAGIndex")
def test_answer_generates_response_with_mistral(
    mock_index_class,
):
    mock_index = mock_index_class.return_value
    mock_index.index = Mock()
    mock_index.search.return_value = [
        make_result()
    ]

    service = EventRAGService()
    service.index = mock_index

    service.client = Mock()

    service.client.chat.complete.return_value = Mock(
        choices=[
            Mock(
                message=Mock(
                    content="Il y a un concert à Paris."
                )
            )
        ]
    )

    result = service.answer("Quel concert ?")

    assert result["used_mistral"] is True
    assert result["answer"] == (
        "Il y a un concert à Paris."
    )

    service.client.chat.complete.assert_called_once()
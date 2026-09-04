"""Tests de l'évaluation Ragas."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.evaluation.ragas_evaluation import (
    _build_ragas_dataset,
    run_ragas_evaluation,
)


def make_service():
    service = Mock()

    service.answer.return_value = {
        "answer": "Il y a un concert.",
        "context": [
            {
                "score": 0.9,
                "document": Document(
                    page_content="Concert à Paris le 10 août.",
                    metadata={"uid": "event-1"},
                ),
            }
        ],
        "used_mistral": True,
    }

    return service


def test_build_ragas_dataset():
    service = make_service()

    dataset = _build_ragas_dataset(service)

    assert len(dataset) == 12

    service.answer.assert_called()

    for row in dataset:
        assert row.user_input
        assert row.reference
        assert row.response
        assert isinstance(row.retrieved_contexts, list)


def test_build_ragas_dataset_calls_rag_for_each_question():
    service = make_service()

    dataset = _build_ragas_dataset(service)

    assert len(dataset) == service.answer.call_count


def test_build_ragas_dataset_rejects_missing_user_input():
    service = make_service()

    with patch(
        "app.evaluation.ragas_evaluation.load_evaluation_rows",
        return_value=[
            {
                "reference": "Référence",
            }
        ],
    ):
        with pytest.raises(
            ValueError,
            match="user_input",
        ):
            _build_ragas_dataset(service)


def test_build_ragas_dataset_rejects_missing_reference():
    service = make_service()

    with patch(
        "app.evaluation.ragas_evaluation.load_evaluation_rows",
        return_value=[
            {
                "user_input": "Question",
            }
        ],
    ):
        with pytest.raises(
            ValueError,
            match="reference",
        ):
            _build_ragas_dataset(service)


@patch(
    "app.evaluation.ragas_evaluation.ContextRecall"
)
@patch(
    "app.evaluation.ragas_evaluation.ContextPrecision"
)
@patch(
    "app.evaluation.ragas_evaluation.AnswerRelevancy"
)
@patch(
    "app.evaluation.ragas_evaluation.Faithfulness"
)
@patch(
    "app.evaluation.ragas_evaluation.evaluate"
)
@patch(
    "app.evaluation.ragas_evaluation._create_mistral_models"
)
@patch(
    "app.evaluation.ragas_evaluation.EventRAGService"
)
def test_run_ragas_evaluation(
    mock_service_class,
    mock_create_models,
    mock_evaluate,
    mock_faithfulness,
    mock_answer_relevancy,
    mock_context_precision,
    mock_context_recall,
):
    """Teste l'orchestration complète de l'évaluation Ragas."""

    mock_service = mock_service_class.return_value

    mock_service.answer.return_value = {
        "answer": "Réponse test",
        "context": [],
        "used_mistral": False,
    }

    mock_create_models.return_value = (
        Mock(),
        Mock(),
    )

    mock_faithfulness.return_value = Mock()
    mock_answer_relevancy.return_value = Mock()
    mock_context_precision.return_value = Mock()
    mock_context_recall.return_value = Mock()

    mock_result = Mock()
    mock_result.to_dict.return_value = {
        "faithfulness": 0.9,
    }

    mock_evaluate.return_value = mock_result

    with patch(
        "app.evaluation.ragas_evaluation.MISTRAL_API_KEY",
        "fake-key",
    ):
        result = run_ragas_evaluation()

    assert result["status"] == "ok"

    assert result["metrics"] == {
        "faithfulness": 0.9,
    }

    mock_service.answer.assert_called()

    mock_create_models.assert_called_once()

    mock_faithfulness.assert_called_once()
    mock_answer_relevancy.assert_called_once()
    mock_context_precision.assert_called_once()
    mock_context_recall.assert_called_once()

    mock_evaluate.assert_called_once()

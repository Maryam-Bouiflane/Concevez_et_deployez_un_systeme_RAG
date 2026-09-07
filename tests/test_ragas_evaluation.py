import pytest
from unittest.mock import AsyncMock, Mock, patch

from ragas import EvaluationDataset

from app.evaluation.ragas_evaluation import run_ragas_evaluation


@pytest.mark.asyncio
async def test_run_ragas_evaluation():
    """Teste l'évaluation complète sans appeler réellement Mistral."""

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": "Quels concerts sont prévus à Paris ?",
                "response": "Un concert est prévu à Paris.",
                "retrieved_contexts": [
                    "Concert à Paris le 15 août 2026."
                ],
                "reference": "Un concert est prévu à Paris.",
            }
        ]
    )

    # Résultat retourné par chaque métrique.
    score_result = Mock()
    score_result.value = 0.8

    # Mock des quatre métriques.
    faithfulness = Mock()
    faithfulness.ascore = AsyncMock(return_value=score_result)

    answer_relevancy = Mock()
    answer_relevancy.ascore = AsyncMock(return_value=score_result)

    context_precision = Mock()
    context_precision.ascore = AsyncMock(return_value=score_result)

    context_recall = Mock()
    context_recall.ascore = AsyncMock(return_value=score_result)

    with (
        patch(
            "app.evaluation.ragas_evaluation.EventRAGService"
        ),
        patch(
            "app.evaluation.ragas_evaluation._build_ragas_dataset",
            new=AsyncMock(return_value=dataset),
        ),
        patch(
            "app.evaluation.ragas_evaluation._create_mistral_models",
            return_value=(Mock(), Mock()),
        ),
        patch(
            "app.evaluation.ragas_evaluation.Faithfulness",
            return_value=faithfulness,
        ),
        patch(
            "app.evaluation.ragas_evaluation.AnswerRelevancy",
            return_value=answer_relevancy,
        ),
        patch(
            "app.evaluation.ragas_evaluation.ContextPrecision",
            return_value=context_precision,
        ),
        patch(
            "app.evaluation.ragas_evaluation.ContextRecall",
            return_value=context_recall,
        ),
        patch(
            "app.evaluation.ragas_evaluation.MISTRAL_API_KEY",
            "fake-api-key",
        ),
        patch(
            "app.evaluation.ragas_evaluation.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_ragas_evaluation()

    assert result["status"] == "ok"

    assert result["dataset_size"] == 1

    assert result["metrics"] == {
        "faithfulness": 0.8,
        "answer_relevancy": 0.8,
        "context_precision": 0.8,
        "context_recall": 0.8,
    }

    assert result["scores"] == {
        "faithfulness": [0.8],
        "answer_relevancy": [0.8],
        "context_precision": [0.8],
        "context_recall": [0.8],
    }

    faithfulness.ascore.assert_awaited_once_with(
        user_input="Quels concerts sont prévus à Paris ?",
        response="Un concert est prévu à Paris.",
        retrieved_contexts=[
            "Concert à Paris le 15 août 2026."
        ],
    )

    answer_relevancy.ascore.assert_awaited_once_with(
        user_input="Quels concerts sont prévus à Paris ?",
        response="Un concert est prévu à Paris.",
    )

    context_precision.ascore.assert_awaited_once_with(
        user_input="Quels concerts sont prévus à Paris ?",
        reference="Un concert est prévu à Paris.",
        retrieved_contexts=[
            "Concert à Paris le 15 août 2026."
        ],
    )

    context_recall.ascore.assert_awaited_once_with(
        user_input="Quels concerts sont prévus à Paris ?",
        retrieved_contexts=[
            "Concert à Paris le 15 août 2026."
        ],
        reference="Un concert est prévu à Paris.",
    )
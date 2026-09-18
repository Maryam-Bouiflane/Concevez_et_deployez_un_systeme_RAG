import pytest

from app.evaluation.ragas_evaluation import _build_ragas_dataset


@pytest.mark.asyncio
async def test_build_ragas_dataset_requires_reference_contexts(monkeypatch):
    dataset = [{
        "user_input": "Question",
        "response": "Réponse",
        "retrieved_contexts": ["ctx"],
        "reference": "Référence",
    }]

    monkeypatch.setattr(
        "app.evaluation.ragas_evaluation.load_evaluation_rows",
        lambda: dataset,
    )

    with pytest.raises(ValueError, match="reference_contexts"):
        await _build_ragas_dataset(service=None)

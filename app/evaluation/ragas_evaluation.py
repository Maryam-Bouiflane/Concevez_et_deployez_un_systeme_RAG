"""Évaluation du système RAG avec Ragas et Mistral."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mistralai.client import Mistral
from openai import AsyncOpenAI
from ragas import EvaluationDataset
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import MISTRAL_API_KEY
from app.core.rag_service import EventRAGService
from app.evaluation.dataset import load_evaluation_rows


MISTRAL_LLM_MODEL = "mistral-small-latest"
MISTRAL_EMBEDDING_MODEL = "mistral-embed"


class MistralEmbeddings(BaseRagasEmbedding):
    """Adaptateur permettant à Ragas d'utiliser les embeddings Mistral."""

    def __init__(
        self,
        client: Mistral,
        model: str = MISTRAL_EMBEDDING_MODEL,
    ) -> None:
        super().__init__()

        self.client = client
        self.model = model

    def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Transforme un texte en vecteur avec Mistral."""

        response = self.client.embeddings.create(
            model=self.model,
            inputs=[text],
        )

        return response.data[0].embedding

    async def aembed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Version asynchrone de l'embedding Mistral."""

        response = await self.client.embeddings.create_async(
            model=self.model,
            inputs=[text],
        )

        return response.data[0].embedding


def _create_mistral_models() -> tuple[Any, MistralEmbeddings]:
    """Initialise le LLM juge et les embeddings Mistral."""

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "La clé MISTRAL_API_KEY n'est pas configurée."
        )

    # Client OpenAI-compatible utilisé par Ragas pour le LLM juge.
    evaluator_client = AsyncOpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
    )

    evaluator_llm = llm_factory(
        MISTRAL_LLM_MODEL,
        client=evaluator_client,
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
    )

    # Client Mistral natif utilisé pour les embeddings.
    mistral_client = Mistral(
        api_key=MISTRAL_API_KEY,
    )

    # Embeddings Mistral utilisés par les métriques qui en ont besoin.
    evaluator_embeddings = MistralEmbeddings(
        client=mistral_client,
        model=MISTRAL_EMBEDDING_MODEL,
    )

    return evaluator_llm, evaluator_embeddings


async def _build_ragas_dataset(
    service: EventRAGService,
) -> EvaluationDataset:
    """
    Exécute le RAG sur chaque question du dataset rag_evaluation.json
    et construit le dataset utilisé par Ragas.

    Le fichier JSON fournit :
        - user_input
        - reference

    Le système RAG fournit :
        - response
        - retrieved_contexts
    """

    evaluation_rows = load_evaluation_rows()

    if not evaluation_rows:
        raise ValueError(
            "Le dataset d'évaluation est vide."
        )

    ragas_rows: list[dict[str, Any]] = []

    for row in evaluation_rows:
        user_input = row.get("user_input")
        reference = row.get("reference")

        if not user_input:
            raise ValueError(
                "Chaque exemple doit contenir 'user_input'."
            )

        if not reference:
            raise ValueError(
                "Chaque exemple doit contenir 'reference'."
            )

        # Exécute réellement notre système RAG.
        result = service.answer(user_input)

        # Pause pour limiter la fréquence des requêtes vers l'API Mistral.
        await asyncio.sleep(2)

        response = result.get("answer", "")
        context_results = result.get("context", [])

        # Ragas attend une liste de chaînes pour retrieved_contexts.
        retrieved_contexts = [
            item["document"].page_content
            for item in context_results
            if isinstance(item, dict)
            and "document" in item
        ]

        ragas_rows.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            }
        )

    return EvaluationDataset.from_list(ragas_rows)


async def run_ragas_evaluation() -> dict[str, Any]:
    """
    Exécute l'évaluation complète du système RAG avec Ragas.

    Pour chaque question du dataset :
        1. Le RAG génère une réponse.
        2. Les contextes récupérés sont enregistrés.
        3. Les métriques Ragas évaluent le résultat.

    Les métriques utilisées sont :
        - Faithfulness
        - AnswerRelevancy
        - ContextPrecision
        - ContextRecall

    Le résultat contient :
        - le score de chaque métrique pour chaque question ;
        - la moyenne de chaque métrique sur l'ensemble du dataset.
    """

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "La clé MISTRAL_API_KEY n'est pas configurée."
        )

    print("\n========== RAGAS EVALUATION ==========")

    # ------------------------------------------------------------------
    # 1. Création du dataset à partir de notre RAG
    # ------------------------------------------------------------------

    service = EventRAGService()

    start = time.perf_counter()

    dataset = await _build_ragas_dataset(service)

    dataset_time = time.perf_counter() - start

    print(
        f"[1/2] Construction du dataset : "
        f"{dataset_time:.2f} s"
    )

    rows = dataset.to_list()

    print(
        f"      Nombre de questions      : "
        f"{len(rows)}"
    )

    # ------------------------------------------------------------------
    # 2. Création des modèles utilisés par Ragas
    # ------------------------------------------------------------------

    evaluator_llm, evaluator_embeddings = (
        _create_mistral_models()
    )

    # ------------------------------------------------------------------
    # 3. Création des métriques
    # ------------------------------------------------------------------

    faithfulness = Faithfulness(
        llm=evaluator_llm,
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm,
    )

    context_recall = ContextRecall(
        llm=evaluator_llm,
    )

    # ------------------------------------------------------------------
    # 4. Stockage des scores
    # ------------------------------------------------------------------

    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    # ------------------------------------------------------------------
    # 5. Évaluation question par question
    # ------------------------------------------------------------------

    for index, row in enumerate(rows, start=1):

        user_input = row["user_input"]
        response = row["response"]
        reference = row["reference"]
        retrieved_contexts = row["retrieved_contexts"]

        print(
            f"\n---------- QUESTION {index}/{len(rows)} ----------"
        )
        print(f"Question : {user_input}")

        # --------------------------------------------------------------
        # Faithfulness
        # --------------------------------------------------------------

        start = time.perf_counter()

        result = await faithfulness.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )

        elapsed = time.perf_counter() - start

        score = float(result.value)

        scores["faithfulness"].append(score)

        print(
            f"Faithfulness       : {score:.3f} "
            f"({elapsed:.2f} s)"
        )

        await asyncio.sleep(1)

        # --------------------------------------------------------------
        # Answer Relevancy
        # --------------------------------------------------------------

        start = time.perf_counter()

        result = await answer_relevancy.ascore(
            user_input=user_input,
            response=response,
        )

        elapsed = time.perf_counter() - start

        score = float(result.value)

        scores["answer_relevancy"].append(score)

        print(
            f"Answer Relevancy   : {score:.3f} "
            f"({elapsed:.2f} s)"
        )

        await asyncio.sleep(1)

        # --------------------------------------------------------------
        # Context Precision
        # --------------------------------------------------------------

        start = time.perf_counter()

        result = await context_precision.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )

        elapsed = time.perf_counter() - start

        score = float(result.value)

        scores["context_precision"].append(score)

        print(
            f"Context Precision  : {score:.3f} "
            f"({elapsed:.2f} s)"
        )

        await asyncio.sleep(1)

        # --------------------------------------------------------------
        # Context Recall
        # --------------------------------------------------------------

        start = time.perf_counter()

        result = await context_recall.ascore(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )

        elapsed = time.perf_counter() - start

        score = float(result.value)

        scores["context_recall"].append(score)

        print(
            f"Context Recall     : {score:.3f} "
            f"({elapsed:.2f} s)"
        )

        await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # 6. Calcul des moyennes sur l'ensemble du dataset
    # ------------------------------------------------------------------

    metrics_average = {
        metric_name: (
            sum(metric_scores) / len(metric_scores)
            if metric_scores
            else 0.0
        )
        for metric_name, metric_scores in scores.items()
    }

    # ------------------------------------------------------------------
    # 7. Affichage des résultats
    # ------------------------------------------------------------------

    print("\n========== RESULTATS RAGAS ==========")
    print(
        f"Nombre de questions évaluées : {len(rows)}"
    )
    print("-------------------------------------")

    for metric_name, score in metrics_average.items():
        print(
            f"{metric_name:<20}: {score:.3f}"
        )

    print("=====================================\n")

    return {
        "status": "ok",
        "metrics": metrics_average,
        "scores": scores,
        "dataset_size": len(rows),
    }


if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation())
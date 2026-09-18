"""Évaluation du système RAG avec Ragas et Mistral."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

from instructor.v2.core.errors import IncompleteOutputException
from mistralai.client import Mistral
from openai import AsyncOpenAI, RateLimitError
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
from app.config import INDEX_PATH, METADATA_PATH, TOP_K, SIMILARITY_THRESHOLD
from pathlib import Path
import json
from datetime import datetime, timezone 
import hashlib


def compute_system_version(file_paths: list[str], params: dict[str, Any]) -> str:
    """Compute a SHA256 fingerprint for a list of files and parameters.

    Missing files are included by their path string to keep the hash
    stable across environments where some files may be absent.
    """
    h = hashlib.sha256()

    for p in file_paths:
        try:
            data = Path(p).read_bytes()
        except Exception:
            data = p.encode()

        h.update(data)

    # Include params in a deterministic order
    for key in sorted(params.keys()):
        h.update(f"{key}={params[key]}".encode())

    return h.hexdigest()


def _truncate_contexts_for_metric(
    contexts: list[str] | None,
    max_chars: int = 2200,
    max_docs: int = 2,
) -> list[str]:
    """Réduit les contextes à une taille exploitable par les métriques Ragas."""

    if not contexts:
        return []

    safe_contexts: list[str] = []
    for context in contexts[:max_docs]:
        text = " ".join(str(context).split())
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].strip()
            if len(text) < 80:
                text = text[:max_chars]
            text = f"{text}..."
        safe_contexts.append(text)

    return safe_contexts


MISTRAL_LLM_MODEL = "mistral-small-latest"
MISTRAL_EMBEDDING_MODEL = "mistral-embed"

# ----------------------------------------------------------------------
# Configuration du rate limiting
# ----------------------------------------------------------------------

MAX_RETRIES = 5

INITIAL_RETRY_DELAY = 2.0

MAX_RETRY_DELAY = 30.0

# Pause entre deux appels normaux à l'API Mistral.
REQUEST_DELAY = 2.0


T = TypeVar("T")


async def _call_with_retry(
    operation: Callable[[], Awaitable[T]],
    operation_name: str,
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Exécute une opération asynchrone avec retry en cas de rate limit.

    En cas de HTTP 429, le délai augmente progressivement :

        tentative 1 → 2 s
        tentative 2 → 4 s
        tentative 3 → 8 s
        tentative 4 → 16 s
        tentative 5 → 30 s maximum

    Les autres exceptions sont propagées immédiatement.
    """

    for attempt in range(max_retries + 1):

        try:
            return await operation()

        except RateLimitError:

            if attempt >= max_retries:
                print(
                    f"[ERREUR] {operation_name} : "
                    f"limite Mistral toujours atteinte "
                    f"après {max_retries + 1} tentatives."
                )

                raise

            delay = min(
                INITIAL_RETRY_DELAY * (2**attempt),
                MAX_RETRY_DELAY,
            )

            print(
                f"[RATE LIMIT] {operation_name} : "
                f"HTTP 429."
            )

            print(
                f"[RETRY] Nouvelle tentative dans "
                f"{delay:.0f} s "
                f"(tentative {attempt + 1}/{max_retries})."
            )

            await asyncio.sleep(delay)

    raise RuntimeError(
        f"L'opération '{operation_name}' "
        "n'a pas pu être exécutée."
    )


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

    # Client OpenAI-compatible utilisé par Ragas
    # pour appeler le LLM juge Mistral.
    evaluator_client = AsyncOpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
    )

    evaluator_llm = llm_factory(
        MISTRAL_LLM_MODEL,
        client=evaluator_client,
        temperature=0.0,
        top_p=1.0,
        max_tokens=4096,
    )

    # Client Mistral natif utilisé pour les embeddings.
    mistral_client = Mistral(
        api_key=MISTRAL_API_KEY,
    )

    evaluator_embeddings = MistralEmbeddings(
        client=mistral_client,
        model=MISTRAL_EMBEDDING_MODEL,
    )

    return evaluator_llm, evaluator_embeddings


async def _build_ragas_dataset(
    service: EventRAGService,
) -> EvaluationDataset:
    """
    Exécute le RAG sur chaque question du dataset et construit
    le dataset utilisé par Ragas.

    Pour chaque question :

        question
            ↓
        Query Parser
            ↓
        metadata filtering
            ↓
        FAISS
            ↓
        génération Mistral
            ↓
        réponse + contextes
            ↓
        dataset Ragas
    """

    try:
        evaluation_rows = load_evaluation_rows()
    except FileNotFoundError as exc:
        print(
            f"[INFO] Dataset d'évaluation introuvable: {exc}. "
            "Tentative de construction automatique via le script de génération."
        )

        try:
            # Import local builder script and run it to create the file.
            from app.scripts import build_rag_evaluation_dataset as _rag_builder

            _rag_builder.main()

            evaluation_rows = load_evaluation_rows()

        except Exception as build_exc:
            raise RuntimeError(
                "Impossible de construire automatiquement le dataset d'évaluation. "
                "Exécutez 'uv run python -m app.scripts.build_rag_evaluation_dataset' manuellement."
            ) from build_exc

    if not evaluation_rows:
        raise ValueError(
            "Le dataset d'évaluation est vide."
        )

    ragas_rows: list[dict[str, Any]] = []
    updated_rows: list[dict[str, Any]] = []

    total = len(evaluation_rows)

    print(
        f"\n[DATASET] {total} questions à traiter."
    )

    for index, row in enumerate(
        evaluation_rows,
        start=1,
    ):

        user_input = row.get("user_input")
        reference = row.get("reference")

        if not user_input:
            raise ValueError(
                "Chaque exemple doit contenir "
                "'user_input'."
            )

        if not reference:
            raise ValueError(
                "Chaque exemple doit contenir "
                "'reference'."
            )

        if "reference_contexts" not in row:
            raise ValueError(
                "Chaque exemple doit contenir 'reference_contexts'. "
                "Utilisez le script resolve_reference_contexts.py pour le générer."
            )

        print(
            f"\n========== ASK {index}/{total} =========="
        )

        print(
            f"Question : {user_input}"
        )

        # --------------------------------------------------------------
        # Exécution du véritable pipeline RAG.
        # --------------------------------------------------------------

        start = time.perf_counter()

        try:
            result = await _call_with_retry(
                operation=lambda: asyncio.to_thread(
                    service.answer,
                    user_input,
                ),
                operation_name=(
                    f"RAG question {index}/{total}"
                ),
            )

        except RateLimitError as exc:
            raise RuntimeError(
                "L'API Mistral a atteint sa limite de "
                "requêtes pendant la construction du "
                "dataset Ragas."
            ) from exc

        elapsed = time.perf_counter() - start

        print(
            f"[OK] Réponse RAG obtenue en "
            f"{elapsed:.2f} s"
        )

        # Pause entre deux questions.
        await asyncio.sleep(
            REQUEST_DELAY
        )

        response = result.get(
            "answer",
            "",
        )

        context_results = result.get(
            "context",
            [],
        )

        # Ragas attend une liste de chaînes
        # pour retrieved_contexts.
        retrieved_contexts = [
            item["document"].page_content
            for item in context_results
            if (
                isinstance(item, dict)
                and "document" in item
            )
        ]

        print(
            f"[INFO] Contextes récupérés : "
            f"{len(retrieved_contexts)}"
        )

        updated_row = dict(row)
        updated_row["response"] = response
        updated_row["retrieved_contexts"] = retrieved_contexts
        updated_row["reference_contexts"] = list(row.get("reference_contexts") or [])
        updated_rows.append(updated_row)

        ragas_rows.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
                "reference_contexts": list(row.get("reference_contexts") or []),
            }
        )

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "evaluation"
        / "rag_evaluation.json"
    )
    with dataset_path.open("w", encoding="utf-8") as file:
        json.dump(updated_rows, file, ensure_ascii=False, indent=2)

    print(f"[INFO] Dataset d'évaluation mis à jour : {dataset_path}")

    return EvaluationDataset.from_list(
        ragas_rows
    )


async def _score_metric(
    metric: Any,
    metric_name: str,
    operation_kwargs: dict[str, Any],
) -> float:
    """
    Calcule une métrique Ragas avec gestion du rate limit.
    """

    start = time.perf_counter()

    try:
        result = await _call_with_retry(
            operation=lambda: metric.ascore(
                **operation_kwargs
            ),
            operation_name=metric_name,
        )

    except IncompleteOutputException:
        if metric_name == "Faithfulness" and "retrieved_contexts" in operation_kwargs:
            safe_kwargs = dict(operation_kwargs)
            safe_kwargs["retrieved_contexts"] = _truncate_contexts_for_metric(
                operation_kwargs.get("retrieved_contexts", []),
            )
            print(
                "[WARN] Contexte Ragas trop long pour Faithfulness : "
                "nouvel essai avec contexte tronqué."
            )
            result = await _call_with_retry(
                operation=lambda: metric.ascore(
                    **safe_kwargs
                ),
                operation_name=f"{metric_name} (contexte tronqué)",
            )
        else:
            raise

    except TypeError as exc:
        # Compatibility fallback: some metric implementations do not
        # accept ``reference_contexts``. If that kwarg triggered the
        # TypeError, retry without it.
        msg = str(exc)
        if "reference_contexts" in msg or "unexpected keyword" in msg:
            slim_kwargs = {k: v for k, v in operation_kwargs.items() if k != "reference_contexts"}
            try:
                result = await _call_with_retry(
                    operation=lambda: metric.ascore(
                        **slim_kwargs
                    ),
                    operation_name=f"{metric_name} (fallback)",
                )
            except RateLimitError as exc2:
                raise RuntimeError(
                    f"L'API Mistral a atteint sa limite de "
                    f"requêtes pendant le calcul de "
                    f"{metric_name} (fallback)."
                ) from exc2
        else:
            raise

    except RateLimitError as exc:
        raise RuntimeError(
            f"L'API Mistral a atteint sa limite de "
            f"requêtes pendant le calcul de "
            f"{metric_name}."
        ) from exc

    elapsed = time.perf_counter() - start

    score = float(result.value)

    print(
        f"{metric_name:<20}: "
        f"{score:.3f} "
        f"({elapsed:.2f} s)"
    )

    return score


async def run_ragas_evaluation() -> dict[str, Any]:
    """
    Exécute l'évaluation complète du système RAG avec Ragas.

    Pour chaque question du dataset :

        1. Le RAG génère une réponse.
        2. Les contextes récupérés sont enregistrés.
        3. Faithfulness est calculé.
        4. Answer Relevancy est calculé.
        5. Context Precision est calculé.
        6. Context Recall est calculé.

    Le résultat contient :

        - le score de chaque métrique pour chaque question ;
        - la moyenne de chaque métrique ;
        - la taille du dataset.
    """

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "La clé MISTRAL_API_KEY n'est pas configurée."
        )

    print(
        "\n========== RAGAS EVALUATION =========="
    )

    # ------------------------------------------------------------------
    # 1. Création du dataset à partir de notre RAG
    # ------------------------------------------------------------------

    service = EventRAGService()

    start = time.perf_counter()

    # The evaluation endpoint must regenerate the live RAG outputs on each
    # call. We intentionally do not reuse the cached ragas_dataset.json as it
    # can contain stale temporal answers and contexts.
    dataset = await _build_ragas_dataset(service)

    dataset_time = time.perf_counter() - start

    print(
        f"\n[1/2] Construction du dataset : "
        f"{dataset_time:.2f} s"
    )

    rows = dataset.to_list()

    print(
        f"      Nombre de questions : "
        f"{len(rows)}"
    )

    # ------------------------------------------------------------------
    # 2. Création des modèles utilisés par Ragas
    # ------------------------------------------------------------------

    print(
        "\n[2/2] Initialisation des modèles Ragas..."
    )

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

    total = len(rows)

    for index, row in enumerate(
        rows,
        start=1,
    ):

        user_input = row["user_input"]
        response = row["response"]
        reference = row["reference"]
        retrieved_contexts = row[
            "retrieved_contexts"
        ]
        # Prefer explicit reference_contexts if present (list). If not
        # provided, compute an implicit list by splitting `reference` on
        # ';' but do not pass it to metrics (some tests expect no
        # `reference_contexts` kwarg when it wasn't explicit).
        has_explicit_reference_contexts = "reference_contexts" in row
        if has_explicit_reference_contexts:
            reference_contexts = row["reference_contexts"]
        else:
            reference_contexts = [
                ctx.strip() for ctx in reference.split(";") if ctx.strip()
            ]

        print(
            f"\n---------- QUESTION "
            f"{index}/{total} ----------"
        )

        print(
            f"Question : {user_input}"
        )

        # --------------------------------------------------------------
        # Faithfulness
        # --------------------------------------------------------------

        score = await _score_metric(
            metric=faithfulness,
            metric_name="Faithfulness",
            operation_kwargs={
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": (
                    retrieved_contexts
                ),
            },
        )

        scores[
            "faithfulness"
        ].append(score)

        await asyncio.sleep(
            REQUEST_DELAY
        )

        # --------------------------------------------------------------
        # Answer Relevancy
        # --------------------------------------------------------------

        score = await _score_metric(
            metric=answer_relevancy,
            metric_name="Answer Relevancy",
            operation_kwargs={
                "user_input": user_input,
                "response": response,
            },
        )

        scores[
            "answer_relevancy"
        ].append(score)

        await asyncio.sleep(
            REQUEST_DELAY
        )

        # --------------------------------------------------------------
        # Context Precision
        # --------------------------------------------------------------

        cp_kwargs = {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
        }
        if has_explicit_reference_contexts:
            cp_kwargs["reference_contexts"] = reference_contexts

        score = await _score_metric(
            metric=context_precision,
            metric_name="Context Precision",
            operation_kwargs=cp_kwargs,
        )

        scores[
            "context_precision"
        ].append(score)

        await asyncio.sleep(
            REQUEST_DELAY
        )

        # --------------------------------------------------------------
        # Context Recall
        # --------------------------------------------------------------

        cr_kwargs = {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        }
        if has_explicit_reference_contexts:
            cr_kwargs["reference_contexts"] = reference_contexts

        score = await _score_metric(
            metric=context_recall,
            metric_name="Context Recall",
            operation_kwargs=cr_kwargs,
        )

        scores[
            "context_recall"
        ].append(score)

        await asyncio.sleep(
            REQUEST_DELAY
        )

    # ------------------------------------------------------------------
    # 6. Calcul des moyennes
    # ------------------------------------------------------------------

    metrics_average = {
        metric_name: (
            sum(metric_scores)
            / len(metric_scores)
            if metric_scores
            else 0.0
        )
        for metric_name, metric_scores
        in scores.items()
    }

    # ------------------------------------------------------------------
    # 7. Affichage des résultats
    # ------------------------------------------------------------------

    print(
        "\n========== RESULTATS RAGAS =========="
    )

    print(
        f"Nombre de questions évaluées : "
        f"{len(rows)}"
    )

    print(
        "-------------------------------------"
    )

    for metric_name, score in (
        metrics_average.items()
    ):
        print(
            f"{metric_name:<20}: "
            f"{score:.3f}"
        )

    print(
        "=====================================\n"
    )

    return {
        "status": "ok",
        "metrics": metrics_average,
        "scores": scores,
        "dataset_size": len(rows),
    }


if __name__ == "__main__":
    asyncio.run(
        run_ragas_evaluation()
    )
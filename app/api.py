"""API FastAPI pour poser des questions au système RAG."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.rag_service import EventRAGService
from app.evaluation.ragas_evaluation import run_ragas_evaluation


app = FastAPI(
    title="POC RAG Open Agenda",
    version="0.1.0",
)


service = EventRAGService()


class QuestionRequest(BaseModel):
    """Requête POST attendue par l'endpoint /ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question posée par l'utilisateur.",
    )


class QuestionResponse(BaseModel):
    """Réponse retournée par le système RAG."""

    answer: str
    used_mistral: bool
    context: list[dict[str, Any]]


@app.get("/")
def healthcheck() -> dict[str, str]:
    """Vérifie que l'API est disponible."""

    return {
        "status": "ok",
    }


@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    payload: QuestionRequest,
) -> QuestionResponse:
    """Répond à une question avec le système RAG."""

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    try:
        result = service.answer(question)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur est survenue lors du traitement "
                "de la question."
            ),
        ) from exc

    return QuestionResponse(
        **result,
    )


@app.post("/rebuild")
def rebuild_index() -> dict[str, str]:
    """Reconstruit l'index vectoriel depuis Open Agenda."""

    try:
        service.build_index_from_openagenda()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur est survenue lors de la "
                "reconstruction de l'index."
            ),
        ) from exc

    return {
        "status": "index rebuilt",
    }


@app.post("/evaluate")
async def evaluate_rag() -> Any:
    """Lance l'évaluation du système RAG avec Ragas."""

    try:
        return await run_ragas_evaluation()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur est survenue lors de "
                "l'évaluation Ragas."
            ),
        ) from exc
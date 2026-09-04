"""Tests des endpoints FastAPI."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_healthcheck():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


@patch("app.api.service.answer")
def test_ask_returns_answer(mock_answer):
    mock_answer.return_value = {
        "answer": "Il y a un concert.",
        "used_mistral": True,
        "context": [],
    }

    response = client.post(
        "/ask",
        json={"question": "Y a-t-il un concert ?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Il y a un concert."
    assert data["used_mistral"] is True
    assert data["context"] == []

    mock_answer.assert_called_once_with(
        "Y a-t-il un concert ?"
    )


def test_ask_rejects_whitespace_question():
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "La question ne peut pas être vide."
    )


@patch("app.api.service.build_index_from_openagenda")
def test_rebuild_index(mock_build):
    response = client.post("/rebuild")

    assert response.status_code == 200
    assert response.json() == {
        "status": "index rebuilt"
    }

    mock_build.assert_called_once()


@patch("app.api.run_ragas_evaluation")
def test_evaluate(mock_evaluate):
    mock_evaluate.return_value = {
        "status": "ok",
        "metrics": {
            "faithfulness": 0.9,
        },
    }

    response = client.post("/evaluate")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "metrics": {
            "faithfulness": 0.9,
        },
    }

    mock_evaluate.assert_called_once()
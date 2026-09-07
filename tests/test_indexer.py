"""Tests unitaires pour l'indexation vectorielle."""

from unittest.mock import Mock

import numpy as np
import pytest

from app.core.indexer import EventRAGIndex


def make_event(
    uid="event-1",
    text="Titre: Concert\nVille: Paris",
):
    return {
        "uid": uid,
        "canonicalurl": "https://example.com/event",
        "slug": "concert",
        "text": text,
        "location_coordinates.lon": 2.35,
        "location_coordinates.lat": 48.85,
    }


def test_build_documents_ignores_events_without_text():
    indexer = EventRAGIndex()

    events = [
        make_event(text="Concert"),
        make_event(uid="event-2", text=None),
        make_event(uid="event-3", text="Festival"),
    ]

    documents = indexer._build_documents(events)

    assert len(documents) == 2
    assert documents[0].page_content == "Concert"
    assert documents[1].page_content == "Festival"


def test_build_documents_creates_expected_metadata():
    indexer = EventRAGIndex()

    event = make_event()

    documents = indexer._build_documents([event])

    assert len(documents) == 1

    metadata = documents[0].metadata

    assert metadata["uid"] == "event-1"
    assert metadata["canonicalurl"] == "https://example.com/event"
    assert metadata["slug"] == "concert"
    assert metadata["location_coordinates"] == {
        "lon": 2.35,
        "lat": 48.85,
    }


def test_create_embeddings_batches_requests():
    indexer = EventRAGIndex()

    indexer.client.embeddings.create = Mock(
        side_effect=[
            Mock(
                data=[
                    Mock(embedding=[1.0, 2.0]),
                    Mock(embedding=[3.0, 4.0]),
                ]
            ),
            Mock(
                data=[
                    Mock(embedding=[5.0, 6.0]),
                ]
            ),
        ]
    )

    result = indexer._create_embeddings(
        ["text 1", "text 2", "text 3"],
        batch_size=2,
    )

    assert result.dtype == np.float32
    assert result.shape == (3, 2)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    assert indexer.client.embeddings.create.call_count == 2


def test_build_index_rejects_empty_documents():
    indexer = EventRAGIndex()

    with pytest.raises(
        ValueError,
        match="Aucun événement contenant du texte exploitable",
    ):
        indexer.build_index([])


def test_get_embedding_returns_vector():
    indexer = EventRAGIndex()

    indexer.index = Mock()
    indexer.index.ntotal = 2

    indexer.index.reconstruct.return_value = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    result = indexer.get_embedding(0)

    assert result.dtype == np.float32

    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 2.0],
            dtype=np.float32,
        ),
    )

    indexer.index.reconstruct.assert_called_once_with(0)


def test_get_embedding_rejects_negative_index():
    indexer = EventRAGIndex()

    indexer.index = Mock()
    indexer.index.ntotal = 2

    with pytest.raises(IndexError):
        indexer.get_embedding(-1)


def test_get_embedding_rejects_index_out_of_range():
    indexer = EventRAGIndex()

    indexer.index = Mock()
    indexer.index.ntotal = 2

    with pytest.raises(IndexError):
        indexer.get_embedding(2)
"""Tests unitaires pour l'indexation vectorielle."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

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

    response = Mock()
    response.data = [
        Mock(embedding=[1.0, 2.0]),
        Mock(embedding=[3.0, 4.0]),
        Mock(embedding=[5.0, 6.0]),
    ]

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

    assert indexer.client.embeddings.create.call_count == 2


def test_build_index_rejects_empty_documents():
    indexer = EventRAGIndex()

    with pytest.raises(
        ValueError,
        match="Aucun événement contenant du texte exploitable",
    ):
        indexer.build_index([])


def test_search_uses_similarity_threshold():
    indexer = EventRAGIndex()

    indexer.index = Mock()
    indexer.index.ntotal = 2
    
    indexer.index.search.return_value = (
        np.array([[0.9, 0.3]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
    )

    indexer.documents = [
        Document(page_content="Concert à Paris"),
        Document(page_content="Événement éloigné"),
    ]

    indexer._create_embeddings = Mock(
        return_value=np.array(
            [[1.0, 0.0]],
            dtype=np.float32,
        )
    )

    results = indexer.search(
        "concert",
        top_k=2,
        threshold=0.45,
    )

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.9)
    assert (
        results[0]["document"].page_content
        == "Concert à Paris"
    )
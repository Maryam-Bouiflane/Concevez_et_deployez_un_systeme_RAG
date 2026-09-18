import json

from app.scripts.resolve_reference_contexts import enrich_reference_contexts


def test_enrich_reference_contexts_from_doc_ids(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    metadata_path = tmp_path / "metadata.json"

    dataset = [
        {
            "user_input": "Quels événements à Paris ?",
            "reference": "Référence",
            "reference_doc_ids": ["101", "202"],
        },
        {
            "user_input": "Autre question",
            "reference": "Sans ID",
            "reference_doc_ids": [],
        },
    ]
    metadata = [
        {"uid": "101", "text": "Contenu 101"},
        {"uid": "202", "text": "Contenu 202"},
    ]

    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    enriched = enrich_reference_contexts(dataset_path, metadata_path)

    assert enriched[0]["reference_contexts"] == ["Contenu 101", "Contenu 202"]
    assert enriched[1].get("reference_contexts") is None

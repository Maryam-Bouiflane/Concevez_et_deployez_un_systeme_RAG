"""Indexation vectorielle des événements avec Mistral Embeddings et FAISS."""

from __future__ import annotations

import json
from typing import Any

import faiss
import numpy as np

from langchain_core.documents import Document
from mistralai.client import Mistral

from app.config import (
    INDEX_PATH,
    METADATA_PATH,
    MISTRAL_API_KEY,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from pathlib import Path
from datetime import datetime
import hashlib
import json


class EventRAGIndex:
    """
    Gère la vectorisation et le stockage des événements.

    Responsabilités :

        événements
            ↓
        Documents LangChain
            ↓
        embeddings Mistral
            ↓
        index FAISS
            ↓
        sauvegarde / chargement

    La logique de recherche et de filtrage est gérée
    par EventRetriever.
    """

    def __init__(
        self,
        model_name: str = "mistral-embed",
    ) -> None:
        """Initialise le client Mistral et l'index."""

        self.model_name = model_name

        if not MISTRAL_API_KEY:
            raise RuntimeError(
                "La clé MISTRAL_API_KEY n'est pas configurée."
            )

        self.client = Mistral(
            api_key=MISTRAL_API_KEY,
        )

        # Index FAISS.
        #
        # IMPORTANT :
        # self.documents[i] correspond toujours au vecteur
        # FAISS d'indice i.
        self.index: faiss.Index | None = None

        # Documents correspondant aux vecteurs FAISS.
        self.documents: list[Document] = []

        self.metadata_path = METADATA_PATH
        self.index_path = INDEX_PATH

    # ==================================================================
    # DOCUMENTS
    # ==================================================================

    def _build_documents(
        self,
        events: list[dict[str, Any]],
    ) -> list[Document]:
        """
        Construit les Documents LangChain à partir des événements.
        """

        documents: list[Document] = []

        for event in events:
            text = event.get("text")

            # Un événement sans texte exploitable n'est pas indexé.
            if not text:
                continue

            metadata = {
                "uid": event.get("uid"),
                "canonicalurl": event.get("canonicalurl"),
                "slug": event.get("slug"),
                "title": event.get("title_fr"),
                "date_start": event.get("firstdate_begin"),
                "date_end": event.get("lastdate_end"),
                "location_name": event.get("location_name"),
                "location_address": event.get("location_address"),
                "location_city": event.get("location_city"),
                "location_district": event.get(
                    "location_district"
                ),
                "location_postalcode": event.get(
                    "location_postalcode"
                ),
                "location_department": event.get(
                    "location_department"
                ),
                "location_region": event.get(
                    "location_region"
                ),
                "location_countrycode": event.get(
                    "location_countrycode"
                ),
                "country": event.get("country_fr"),
                "location_coordinates": {
                    "lon": event.get(
                        "location_coordinates.lon"
                    ),
                    "lat": event.get(
                        "location_coordinates.lat"
                    ),
                },
            }

            documents.append(
                Document(
                    id=str(event["uid"]),
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents

    # ==================================================================
    # EMBEDDINGS
    # ==================================================================

    def _create_embeddings(
        self,
        texts: list[str],
        batch_size: int = 50,
    ) -> np.ndarray:
        """Crée les embeddings Mistral par lots."""

        if not texts:
            return np.empty(
                (0, 0),
                dtype="float32",
            )

        all_embeddings: list[list[float]] = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                start:start + batch_size
            ]

            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=batch,
            )

            all_embeddings.extend(
                item.embedding
                for item in response.data
            )

        return np.asarray(
            all_embeddings,
            dtype="float32",
        )

    # ==================================================================
    # BUILD INDEX
    # ==================================================================

    def build_index(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        """Crée l'index FAISS et sauvegarde les documents associés."""

        self.documents = self._build_documents(events)

        if not self.documents:
            raise ValueError(
                "Aucun événement contenant du texte exploitable "
                "à indexer."
            )

        texts = [
            document.page_content
            for document in self.documents
        ]

        embeddings = self._create_embeddings(texts)

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] == 0
        ):
            raise RuntimeError(
                "Aucun embedding valide n'a été généré."
            )

        dimension = embeddings.shape[1]

        # Produit scalaire.
        #
        # Les embeddings Mistral sont normalisés.
        # Le produit scalaire correspond donc à la
        # similarité cosinus.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        if self.index.ntotal != len(
            self.documents
        ):
            raise RuntimeError(
                "Le nombre de vecteurs FAISS ne correspond "
                "pas au nombre de documents."
            )

        # Création des dossiers.
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Sauvegarde FAISS
        # --------------------------------------------------------------

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        # --------------------------------------------------------------
        # Sauvegarde des documents / metadata
        # --------------------------------------------------------------

        events_to_save = [
            {
                "uid": document.metadata.get("uid"),
                "canonicalurl": document.metadata.get(
                    "canonicalurl"
                ),
                "slug": document.metadata.get("slug"),
                "text": document.page_content,
                "title_fr": document.metadata.get("title"),
                "firstdate_begin": document.metadata.get(
                    "date_start"
                ),
                "lastdate_end": document.metadata.get(
                    "date_end"
                ),
                "location_name": document.metadata.get(
                    "location_name"
                ),
                "location_address": document.metadata.get(
                    "location_address"
                ),
                "location_city": document.metadata.get(
                    "location_city"
                ),
                "location_district": document.metadata.get(
                    "location_district"
                ),
                "location_postalcode": document.metadata.get(
                    "location_postalcode"
                ),
                "location_department": document.metadata.get(
                    "location_department"
                ),
                "location_region": document.metadata.get(
                    "location_region"
                ),
                "location_countrycode": document.metadata.get(
                    "location_countrycode"
                ),
                "country_fr": document.metadata.get("country"),
                "location_coordinates.lon": (
                    document.metadata
                    .get(
                        "location_coordinates",
                        {},
                    )
                    .get("lon")
                ),
                "location_coordinates.lat": (
                    document.metadata
                    .get(
                        "location_coordinates",
                        {},
                    )
                    .get("lat")
                ),
            }
            for document in self.documents
        ]

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                events_to_save,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        # ------------------------------------------------------------------
        # Write a manifest describing the index and metadata state so
        # external tools (evaluation) can decide whether to reuse cached
        # datasets.
        # ------------------------------------------------------------------
        try:
            manifest = {}

            def _file_mtime(path: Path) -> float | None:
                try:
                    return float(path.stat().st_mtime)
                except Exception:
                    return None

            manifest["created_at"] = datetime.utcnow().isoformat()
            manifest["index_path"] = str(self.index_path)
            manifest["metadata_path"] = str(self.metadata_path)
            manifest["index_mtime"] = _file_mtime(Path(self.index_path))
            manifest["metadata_mtime"] = _file_mtime(Path(self.metadata_path))

            # compute a content hash for metadata as index_version
            try:
                metadata_bytes = Path(self.metadata_path).read_bytes()
                index_version = hashlib.sha256(metadata_bytes).hexdigest()
            except Exception:
                index_version = None

            manifest["index_version"] = index_version
            manifest["top_k"] = TOP_K
            manifest["threshold"] = SIMILARITY_THRESHOLD

            manifest_path = self.index_path.parent / "index_manifest.json"
            with manifest_path.open("w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)

        except Exception:
            # Non fatal: manifest best-effort
            pass

    # ==================================================================
    # LOAD
    # ==================================================================

    def load(self) -> None:
        """Recharge l'index FAISS et les documents."""

        if (
            not self.index_path.exists()
            or not self.metadata_path.exists()
        ):
            raise FileNotFoundError(
                "Aucun index vectoriel n'a encore été créé."
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            events = json.load(handle)

        self.documents = self._build_documents(
            events
        )

        if self.index.ntotal != len(
            self.documents
        ):
            raise RuntimeError(
                "Le nombre de vecteurs FAISS ne correspond "
                "pas au nombre de documents chargés."
            )

    # ==================================================================
    # QUERY EMBEDDING
    # ==================================================================

    def create_query_embedding(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Crée l'embedding d'une requête.

        Cette méthode est utilisée par le Retriever.
        """

        if not query.strip():
            raise ValueError(
                "La requête de recherche ne peut pas être vide."
            )

        embedding = self._create_embeddings(
            [query]
        )

        if (
            embedding.ndim != 2
            or embedding.shape[0] == 0
        ):
            raise RuntimeError(
                "Aucun embedding valide n'a été généré "
                "pour la requête."
            )

        return embedding

    # ==================================================================
    # VECTOR ACCESS
    # ==================================================================

    def get_embedding(
        self,
        index: int,
    ) -> np.ndarray:
        """
        Retourne le vecteur FAISS correspondant à un document.

        Le Retriever utilise cette méthode pour reconstruire
        un index FAISS temporaire après filtrage metadata.
        """

        if self.index is None:
            raise RuntimeError(
                "L'index FAISS n'est pas disponible."
            )

        if index < 0 or index >= self.index.ntotal:
            raise IndexError(
                f"Indice FAISS invalide : {index}"
            )

        return self.index.reconstruct(
            int(index)
        )
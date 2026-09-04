"""Indexation vectorielle des événements avec Mistral Embeddings et FAISS."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, TYPE_CHECKING

import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from mistralai.client import Mistral

from app.config import (
    INDEX_PATH,
    METADATA_PATH,
    MISTRAL_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.schemas.search import EventSearchFilters


if TYPE_CHECKING:
    from app.core.query_parser import EventQueryParser


class EventRAGIndex:
    """Gère la vectorisation et la recherche sémantique des événements."""

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

        # Index FAISS contenant les embeddings.
        self.index: faiss.Index | None = None

        # Documents correspondant aux vecteurs FAISS.
        #
        # IMPORTANT :
        # self.documents[i] correspond toujours au vecteur
        # FAISS d'indice i.
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
        """Construit les Documents LangChain à partir des événements."""

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
        """Crée l'index FAISS et sauvegarde les événements associés."""

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
                "uid": document.metadata.get(
                    "uid"
                ),
                "canonicalurl": document.metadata.get(
                    "canonicalurl"
                ),
                "slug": document.metadata.get(
                    "slug"
                ),
                "text": document.page_content,
                "title_fr": document.metadata.get(
                    "title"
                ),
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
                "country_fr": document.metadata.get(
                    "country"
                ),
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
    # METADATA FILTERING
    # ==================================================================

    @staticmethod
    def _normalize_value(
        value: Any,
    ) -> str | None:
        """Normalise une valeur metadata pour les comparaisons."""

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value.casefold()

    @staticmethod
    def _parse_event_date(
        value: Any,
    ) -> date | None:
        """
        Convertit une date OpenAgenda en objet date.

        Accepte notamment :
        - YYYY-MM-DD
        - YYYY-MM-DDTHH:MM:SS
        - YYYY-MM-DDTHH:MM:SS+00:00
        """

        if value is None:
            return None

        if isinstance(value, date):
            return value

        value_str = str(value).strip()

        if not value_str:
            return None

        try:
            return date.fromisoformat(
                value_str[:10]
            )
        except ValueError:
            return None

    @classmethod
    def _event_matches_filters(
        cls,
        document: Document,
        filters: EventSearchFilters,
    ) -> bool:
        """
        Vérifie si un document respecte tous les filtres metadata.

        Les différents filtres sont combinés avec AND.
        """

        metadata = document.metadata

        # --------------------------------------------------------------
        # DATE
        # --------------------------------------------------------------

        if (
            filters.date_from is not None
            or filters.date_to is not None
        ):
            event_start = cls._parse_event_date(
                metadata.get("date_start")
            )

            event_end = cls._parse_event_date(
                metadata.get("date_end")
            )

            # Si aucune date exploitable n'est disponible,
            # l'événement ne peut pas satisfaire une contrainte
            # temporelle.
            if event_start is None:
                return False

            # Un événement ponctuel n'ayant pas de date_end
            # est considéré comme ayant lieu à date_start.
            if event_end is None:
                event_end = event_start

            filter_start = filters.date_from
            filter_end = filters.date_to

            # Une seule borne peut être présente.
            if filter_start is not None:
                if event_end < filter_start:
                    return False

            if filter_end is not None:
                if event_start > filter_end:
                    return False

        # --------------------------------------------------------------
        # GEOGRAPHIE
        # --------------------------------------------------------------

        metadata_filters = {
            "location_city": filters.location_city,
            "location_district": filters.location_district,
            "location_postalcode": filters.location_postalcode,
            "location_department": filters.location_department,
            "location_region": filters.location_region,
            "location_countrycode": filters.location_countrycode,
            "country": filters.country_fr,
        }

        for metadata_key, filter_value in metadata_filters.items():
            if filter_value is None:
                continue

            document_value = cls._normalize_value(
                metadata.get(metadata_key)
            )

            expected_value = cls._normalize_value(
                filter_value
            )

            if document_value != expected_value:
                return False

        return True

    def _get_matching_document_indices(
        self,
        filters: EventSearchFilters,
    ) -> list[int]:
        """Retourne les indices FAISS des documents compatibles."""

        return [
            index
            for index, document in enumerate(
                self.documents
            )
            if self._event_matches_filters(
                document,
                filters,
            )
        ]

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filters: EventSearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche les événements les plus proches.

        Pipeline :

            filters metadata
                    ↓
            candidats compatibles
                    ↓
            embedding de la question complète
                    ↓
            FAISS
                    ↓
            threshold
                    ↓
            top_k

        Si filters est None ou vide :

            question complète
                    ↓
                  FAISS
                    ↓
                threshold
                    ↓
                  top_k
        """

        if not query.strip():
            raise ValueError(
                "La requête de recherche ne peut pas être vide."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k doit être supérieur à zéro."
            )

        if self.index is None:
            self.load()

        if self.index is None:
            raise RuntimeError(
                "L'index FAISS n'est pas disponible."
            )

        if self.index.ntotal == 0:
            return []

        # --------------------------------------------------------------
        # 1. Déterminer les candidats metadata
        # --------------------------------------------------------------

        candidate_indices: list[int] | None = None

        if filters is not None:
            candidate_indices = (
                self._get_matching_document_indices(
                    filters
                )
            )

            # Aucun document ne respecte les filtres.
            if not candidate_indices:
                return []

        # --------------------------------------------------------------
        # 2. Embedding de la QUESTION COMPLÈTE
        # --------------------------------------------------------------

        embedding = self._create_embeddings(
            [query]
        )

        if embedding.shape[0] == 0:
            return []

        # --------------------------------------------------------------
        # 3. Recherche FAISS
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Cas 1 : aucun filtre metadata.
        #
        # On interroge directement l'index FAISS complet.
        # --------------------------------------------------------------

        if candidate_indices is None:
            search_k = min(
                top_k,
                self.index.ntotal,
            )

            if search_k == 0:
                return []

            scores, indices = self.index.search(
                embedding,
                search_k,
            )

            results: list[dict[str, Any]] = []

            for score, index in zip(
                scores[0],
                indices[0],
            ):
                if index < 0:
                    continue

                similarity = float(score)

                if similarity < threshold:
                    continue

                document = self.documents[
                    int(index)
                ]

                results.append(
                    {
                        "score": similarity,
                        "document": document,
                    }
                )

            return results

        # --------------------------------------------------------------
        # Cas 2 : filtres metadata.
        #
        # FAISS ne sait pas directement filtrer les metadata
        # de nos Documents.
        #
        # On reconstruit donc un petit index FAISS temporaire
        # uniquement avec les vecteurs des candidats compatibles.
        # --------------------------------------------------------------

        candidate_embeddings = np.asarray(
            [
                self.index.reconstruct(
                    int(index)
                )
                for index in candidate_indices
            ],
            dtype="float32",
        )

        if candidate_embeddings.ndim != 2:
            return []

        if candidate_embeddings.shape[0] == 0:
            return []

        filtered_index = faiss.IndexFlatIP(
            candidate_embeddings.shape[1]
        )

        filtered_index.add(
            candidate_embeddings
        )

        search_k = min(
            top_k,
            len(candidate_indices),
        )

        scores, local_indices = filtered_index.search(
            embedding,
            search_k,
        )

        results = []

        for score, local_index in zip(
            scores[0],
            local_indices[0],
        ):
            if local_index < 0:
                continue

            similarity = float(score)

            if similarity < threshold:
                continue

            original_index = candidate_indices[
                int(local_index)
            ]

            document = self.documents[
                original_index
            ]

            results.append(
                {
                    "score": similarity,
                    "document": document,
                }
            )

        return results

    # ==================================================================
    # LANGCHAIN RETRIEVER
    # ==================================================================

    def as_retriever(
        self,
        query_parser: EventQueryParser | None = None,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> BaseRetriever:
        """
        Expose la recherche comme Retriever LangChain.

        LangChain appelle ce retriever avec la question complète.

        Le retriever :

            question complète
                    ↓
            Query Parser
                    ↓
            metadata filters
                    ↓
            index.search(
                question complète,
                filters
            )
                    ↓
            Documents
        """

        index = self

        class FAISSRetriever(BaseRetriever):
            top_k: int
            threshold: float
            query_parser: Any = None

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Any = None,
            ) -> list[Document]:
                """Récupère les documents pertinents."""

                filters: EventSearchFilters | None = None

                # ------------------------------------------------------
                # Query Parser
                # ------------------------------------------------------

                if self.query_parser is not None:
                    parsed_query = (
                        self.query_parser.parse(query)
                    )

                    filters = parsed_query.filters

                # ------------------------------------------------------
                # Recherche
                # ------------------------------------------------------

                results = index.search(
                    query=query,
                    top_k=self.top_k,
                    threshold=self.threshold,
                    filters=filters,
                )

                return [
                    result["document"]
                    for result in results
                ]

        return FAISSRetriever(
            top_k=top_k,
            threshold=threshold,
            query_parser=query_parser,
        )
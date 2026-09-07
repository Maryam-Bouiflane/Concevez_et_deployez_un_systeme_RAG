"""Retriever LangChain avec filtrage metadata et recherche FAISS."""

from __future__ import annotations

from datetime import date
from typing import Any

import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import (
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.core.indexer import EventRAGIndex
from app.core.query_parser import EventQueryParser
from app.schemas.search import EventSearchFilters


class EventRetriever(BaseRetriever):
    """
    Retriever LangChain du système RAG.

    Pipeline :

        question complète
                ↓
          Query Parser
                ↓
        metadata filters
                ↓
        pré-filtrage metadata
                ↓
        embedding question complète
                ↓
        FAISS
                ↓
           threshold
                ↓
             top_k
                ↓
           Documents
    """

    index: Any
    query_parser: Any
    top_k: int = TOP_K
    threshold: float = SIMILARITY_THRESHOLD

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        Récupère les documents pertinents pour une question.

        La question complète est conservée pour la recherche
        sémantique.
        """

        if not query.strip():
            raise ValueError(
                "La requête de recherche ne peut pas être vide."
            )

        # --------------------------------------------------------------
        # 1. Query Parser
        # --------------------------------------------------------------

        parsed_query = self.query_parser.parse(
            query
        )

        filters = parsed_query.filters

        # --------------------------------------------------------------
        # 2. Recherche
        # --------------------------------------------------------------

        results = self._search(
            query=query,
            filters=filters,
        )

        return [
            result["document"]
            for result in results
        ]

    # ==================================================================
    # SEARCH
    # ==================================================================

    def _search(
        self,
        query: str,
        filters: EventSearchFilters | None,
    ) -> list[dict[str, Any]]:
        """
        Effectue la recherche vectorielle après filtrage metadata.

        Les filtres sont appliqués AVANT la recherche FAISS.
        """

        if self.index.index is None:
            self.index.load()

        if self.index.index is None:
            raise RuntimeError(
                "L'index FAISS n'est pas disponible."
            )

        if self.index.index.ntotal == 0:
            return []

        # --------------------------------------------------------------
        # 1. Candidats metadata
        # --------------------------------------------------------------

        candidate_indices: list[int] | None = None

        if filters is not None:
            candidate_indices = (
                self._get_matching_document_indices(
                    filters
                )
            )

            if not candidate_indices:
                return []

        # --------------------------------------------------------------
        # 2. Embedding de la QUESTION COMPLÈTE
        # --------------------------------------------------------------

        query_embedding = (
            self.index.create_query_embedding(
                query
            )
        )

        # --------------------------------------------------------------
        # 3. Aucun filtre metadata
        # --------------------------------------------------------------

        if candidate_indices is None:
            search_k = min(
                self.top_k,
                self.index.index.ntotal,
            )

            if search_k == 0:
                return []

            scores, indices = self.index.index.search(
                query_embedding,
                search_k,
            )

            return self._build_results(
                scores=scores[0],
                indices=indices[0],
            )

        # --------------------------------------------------------------
        # 4. Filtres metadata
        # --------------------------------------------------------------

        candidate_embeddings = np.asarray(
            [
                self.index.get_embedding(
                    index
                )
                for index in candidate_indices
            ],
            dtype="float32",
        )

        if (
            candidate_embeddings.ndim != 2
            or candidate_embeddings.shape[0] == 0
        ):
            return []

        filtered_index = faiss.IndexFlatIP(
            candidate_embeddings.shape[1]
        )

        filtered_index.add(
            candidate_embeddings
        )

        search_k = min(
            self.top_k,
            len(candidate_indices),
        )

        scores, local_indices = filtered_index.search(
            query_embedding,
            search_k,
        )

        results: list[dict[str, Any]] = []

        for score, local_index in zip(
            scores[0],
            local_indices[0],
        ):
            if local_index < 0:
                continue

            similarity = float(score)

            if similarity < self.threshold:
                continue

            original_index = candidate_indices[
                int(local_index)
            ]

            document = self.index.documents[
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
    # RESULTS
    # ==================================================================

    def _build_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Transforme les résultats FAISS en résultats métier."""

        results: list[dict[str, Any]] = []

        for score, index in zip(
            scores,
            indices,
        ):
            if index < 0:
                continue

            similarity = float(score)

            if similarity < self.threshold:
                continue

            document = self.index.documents[
                int(index)
            ]

            results.append(
                {
                    "score": similarity,
                    "document": document,
                }
            )

        return results

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
        Vérifie qu'un document respecte tous les filtres.

        Les filtres sont combinés avec AND.
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

            if event_start is None:
                return False

            # Événement ponctuel.
            if event_end is None:
                event_end = event_start

            filter_start = filters.date_from
            filter_end = filters.date_to

            # Chevauchement entre la période de l'événement
            # et la période recherchée.
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
                self.index.documents
            )
            if self._event_matches_filters(
                document,
                filters,
            )
        ]
"""Service RAG principal utilisant LangChain, FAISS et Mistral."""

from __future__ import annotations

import time
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import (
    MISTRAL_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.core.data_loader import fetch_events_from_openagenda
from app.core.indexer import EventRAGIndex
from app.core.query_parser import EventQueryParser


MISTRAL_LLM_MODEL = "mistral-small-latest"


class EventRAGService:
    """
    Service RAG principal.

    Pipeline :

        Question utilisateur
                │
                ▼
          Query Parser
                │
                ▼
        Metadata filters
                │
                ▼
      Pré-filtrage metadata
                │
                ▼
       FAISS avec la question
          complète
                │
                ▼
            threshold
                │
                ▼
              top_k
                │
                ▼
            Documents
                │
                ▼
              Mistral
                │
                ▼
             Réponse
    """

    def __init__(self) -> None:
        """Initialise le pipeline RAG."""

        self.index = EventRAGIndex()

        self.llm = None
        self.query_parser = None
        self.retriever = None
        self.rag_chain = None
        self._document_chain = None

        # --------------------------------------------------------------
        # LLM Mistral
        # --------------------------------------------------------------

        if MISTRAL_API_KEY:
            self.llm = ChatOpenAI(
                model=MISTRAL_LLM_MODEL,
                api_key=MISTRAL_API_KEY,
                base_url="https://api.mistral.ai/v1",
                temperature=0.0,
                max_tokens=2048,
            )

        self._build_langchain_components()

    # ==================================================================
    # LANGCHAIN
    # ==================================================================

    def _build_langchain_components(self) -> None:
        """Construit les composants LangChain."""

        if self.llm is None:
            return

        # --------------------------------------------------------------
        # Query Parser
        # --------------------------------------------------------------

        self.query_parser = EventQueryParser(
            self.llm
        )

        # --------------------------------------------------------------
        # Prompt de génération
        # --------------------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Tu es un assistant spécialisé dans les événements locaux.

Réponds uniquement à partir des informations présentes
dans le contexte fourni.

Règles importantes :

- N'invente aucune information.
- Si le contexte ne permet pas de répondre correctement,
  indique clairement que l'information n'est pas disponible.
- Réponds en français.
- Sois clair, naturel et concis.
- Lorsque plusieurs événements sont pertinents,
  présente-les de manière structurée.
- Utilise les informations disponibles dans les documents,
  notamment le nom, la date et le lieu lorsqu'ils sont présents.

Le contexte contient les événements récupérés par le système
de recherche. Les événements présents dans ce contexte sont
les seuls événements que tu peux utiliser pour répondre.

Contexte :
{context}
""".strip(),
                ),
                (
                    "human",
                    "{input}",
                ),
            ]
        )

        self._document_chain = (
            create_stuff_documents_chain(
                self.llm,
                prompt,
                output_parser=StrOutputParser(),
            )
        )

    # ==================================================================
    # INDEX
    # ==================================================================

    def build_index_from_openagenda(self) -> None:
        """Récupère les événements et construit l'index."""

        build_start = time.perf_counter()

        print("\n========== BUILD INDEX ==========")

        # --------------------------------------------------------------
        # 1. Récupération OpenAgenda
        # --------------------------------------------------------------

        fetch_start = time.perf_counter()

        events = fetch_events_from_openagenda()

        fetch_time = (
            time.perf_counter()
            - fetch_start
        )

        print(
            f"[1/2] Récupération OpenAgenda   : "
            f"{fetch_time:.2f} s"
        )

        print(
            f"      Événements récupérés      : "
            f"{len(events)}"
        )

        if not events:
            raise RuntimeError(
                "Aucun événement n'a pu être récupéré "
                "depuis Open Agenda."
            )

        # --------------------------------------------------------------
        # 2. Embeddings + FAISS
        # --------------------------------------------------------------

        index_start = time.perf_counter()

        self.index.build_index(events)

        index_time = (
            time.perf_counter()
            - index_start
        )

        total_time = (
            time.perf_counter()
            - build_start
        )

        print(
            f"[2/2] Embeddings + index FAISS : "
            f"{index_time:.2f} s"
        )

        print(
            f"      Documents indexés        : "
            f"{len(self.index.documents)}"
        )

        print("--------------------------------")

        print(
            f"TOTAL BUILD INDEX             : "
            f"{total_time:.2f} s"
        )

        print("================================\n")

        # Le retriever utilise le nouvel index.
        self._create_retriever()

    def _ensure_index_loaded(self) -> None:
        """Charge l'index depuis le disque ou le reconstruit."""

        # --------------------------------------------------------------
        # Index déjà chargé
        # --------------------------------------------------------------

        if self.index.index is not None:
            self._create_retriever()
            return

        # --------------------------------------------------------------
        # Index sauvegardé disponible
        # --------------------------------------------------------------

        if (
            self.index.index_path.exists()
            and self.index.metadata_path.exists()
        ):
            print(
                "[INFO] Index absent en mémoire → "
                "chargement depuis le disque"
            )

            load_start = time.perf_counter()

            self.index.load()

            load_time = (
                time.perf_counter()
                - load_start
            )

            print(
                f"[INFO] Index chargé en "
                f"{load_time:.2f} s"
            )

            print(
                f"[INFO] Documents chargés : "
                f"{len(self.index.documents)}"
            )

            self._create_retriever()

            return

        # --------------------------------------------------------------
        # Aucun index
        # --------------------------------------------------------------

        print(
            "[INFO] Aucun index disponible → "
            "reconstruction depuis OpenAgenda"
        )

        self.build_index_from_openagenda()

    # ==================================================================
    # RETRIEVER
    # ==================================================================

    def _create_retriever(self) -> None:
        """Crée le retriever LangChain."""

        self.retriever = self.index.as_retriever(
            query_parser=self.query_parser,
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )

        if self._document_chain is not None:
            self.rag_chain = (
                create_retrieval_chain(
                    self.retriever,
                    self._document_chain,
                )
            )

    # ==================================================================
    # ANSWER
    # ==================================================================

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Répond à une question avec le pipeline RAG.

        La question originale complète est conservée
        pour la recherche sémantique FAISS.

        Le Query Parser extrait uniquement les filtres
        metadata.
        """

        if not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        total_start = time.perf_counter()

        print("\n========== ASK ==========")

        # --------------------------------------------------------------
        # 1. Chargement de l'index
        # --------------------------------------------------------------

        index_start = time.perf_counter()

        self._ensure_index_loaded()

        index_time = (
            time.perf_counter()
            - index_start
        )

        print(
            f"[INFO] Vérification/chargement index : "
            f"{index_time:.2f} s"
        )

        # --------------------------------------------------------------
        # 2. Mistral / LangChain indisponible
        # --------------------------------------------------------------

        if (
            self.llm is None
            or self.rag_chain is None
        ):
            print(
                "[2/2] Génération Mistral : "
                "NON DISPONIBLE"
            )

            # Le Query Parser peut néanmoins être utilisé
            # indépendamment de la génération.
            filters = None

            if self.query_parser is not None:
                parsed_query = (
                    self.query_parser.parse(
                        question
                    )
                )

                filters = parsed_query.filters

            # IMPORTANT :
            # on envoie toujours la QUESTION COMPLÈTE
            # à FAISS.
            results = self.index.search(
                query=question,
                top_k=TOP_K,
                threshold=SIMILARITY_THRESHOLD,
                filters=filters,
            )

            total_time = (
                time.perf_counter()
                - total_start
            )

            print(
                f"TOTAL ASK                     : "
                f"{total_time:.2f} s"
            )

            print("=============================\n")

            return {
                "answer": (
                    "La génération de réponse avec Mistral "
                    "n'est pas disponible."
                ),
                "context": results,
                "used_mistral": False,
            }

        # --------------------------------------------------------------
        # 3. Pipeline LangChain complet
        # --------------------------------------------------------------

        mistral_start = time.perf_counter()

        chain_result = self.rag_chain.invoke(
            {
                "input": question,
            }
        )

        mistral_time = (
            time.perf_counter()
            - mistral_start
        )

        # --------------------------------------------------------------
        # 4. Documents récupérés
        # --------------------------------------------------------------

        retrieved_documents = (
            chain_result.get(
                "context",
                [],
            )
        )

        context_results: list[
            dict[str, Any]
        ] = []

        for document in retrieved_documents:
            context_results.append(
                {
                    "score": None,
                    "document": document,
                }
            )

        # --------------------------------------------------------------
        # 5. Réponse
        # --------------------------------------------------------------

        answer_text = chain_result.get(
            "answer",
            "",
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[2/2] Génération LangChain/Mistral : "
            f"{mistral_time:.2f} s"
        )

        print(
            f"      Documents récupérés       : "
            f"{len(retrieved_documents)}"
        )

        print("--------------------------------")

        print(
            f"TOTAL ASK                     : "
            f"{total_time:.2f} s"
        )

        print("=============================\n")

        return {
            "answer": answer_text,
            "context": context_results,
            "used_mistral": True,
        }
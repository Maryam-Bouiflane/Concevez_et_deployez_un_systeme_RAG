"""Service RAG principal utilisant LangChain, FAISS et Mistral."""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from typing import Any

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from zoneinfo import ZoneInfo

from app.config import (
    MISTRAL_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.core.data_loader import fetch_events_from_openagenda
from app.core.indexer import EventRAGIndex
from app.core.query_parser import EventQueryParser
from app.core.retriever import EventRetriever


MISTRAL_LLM_MODEL = "mistral-small-latest"
PARIS_TZ = ZoneInfo("Europe/Paris")

class EventRAGService:
    """
    Service principal du système RAG.

    Responsabilités :

        - initialiser les composants ;
        - construire / charger l'index ;
        - assembler le Retriever et la chaîne de génération ;
        - lancer la génération de réponse.

    Le Retriever est responsable de :

        - analyser la question ;
        - extraire les filtres metadata ;
        - appliquer les filtres ;
        - effectuer la recherche FAISS ;
        - appliquer le seuil de similarité ;
        - appliquer TOP_K ;
        - retourner les documents avec leurs scores.

    Les scores de similarité sont utilisés uniquement par le service
    pour la réponse API. Ils ne sont jamais transmis au LLM.
    """

    @staticmethod
    def _extract_date_mentions(text: str) -> set[str]:
        """Extrait les dates explicites présentes dans un texte."""

        if not text:
            return set()

        pattern = re.compile(
            r"\b(?:"
            r"\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
            r"août|septembre|octobre|novembre|décembre)\s+\d{4}|"
            r"\d{1,2}/\d{1,2}/\d{2,4}|"
            r"\d{4}-\d{2}-\d{2}|"
            r"\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
            r"août|septembre|octobre|novembre|décembre)"
            r")\b",
            re.IGNORECASE,
        )

        matches: set[str] = set()

        for match in pattern.findall(text):
            normalized = " ".join(str(match).strip().split())

            if normalized:
                matches.add(normalized.casefold())

        return matches

    @staticmethod
    def _coerce_text_date(value: str) -> date | None:
        """Convertit une date textuelle en date Python si possible."""

        value = (value or "").strip()

        if not value:
            return None

        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass

        normalized = value.replace("’", "'")

        month_map = {
            "janvier": "01",
            "février": "02",
            "fevrier": "02",
            "mars": "03",
            "avril": "04",
            "mai": "05",
            "juin": "06",
            "juillet": "07",
            "août": "08",
            "aout": "08",
            "septembre": "09",
            "octobre": "10",
            "novembre": "11",
            "décembre": "12",
            "decembre": "12",
        }

        for pattern in (
            r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$",
            r"^(\d{1,2})\s+"
            r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|"
            r"août|aout|septembre|octobre|novembre|décembre|decembre)"
            r"\s+(\d{4})$",
            r"^(\d{1,2})\s+"
            r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|"
            r"août|aout|septembre|octobre|novembre|décembre|decembre)$",
        ):
            match = re.match(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            if pattern.startswith(
                r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$"
            ):
                day, month, year = match.groups()

                try:
                    return date(
                        int(year),
                        int(month),
                        int(day),
                    )
                except ValueError:
                    continue

            day = match.group(1)
            month_name = match.group(2).lower()
            month = month_map.get(month_name)

            if month is None:
                continue

            year = (
                match.group(3)
                if len(match.groups()) >= 3
                else None
            )

            if year is None:
                try:
                    return date(
                        datetime.now().year,
                        int(month),
                        int(day),
                    )
                except ValueError:
                    continue

            try:
                return date(
                    int(year),
                    int(month),
                    int(day),
                )
            except ValueError:
                continue

        for fmt in (
            "%d/%m/%Y",
            "%d/%m/%y",
            "%d %B %Y",
            "%d %B",
        ):
            try:
                return datetime.strptime(
                    value,
                    fmt,
                ).date()
            except ValueError:
                continue

        return None

    def _sanitize_answer_to_context(
        self,
        answer: str,
        context_documents: list[Any],
        parsed_query: Any = None,
    ) -> str:
        """
        Vérifie qu'une date explicite de la réponse respecte
        la période demandée.

        Le Query Parser n'est PAS rappelé ici.
        Le résultat déjà obtenu pendant le retrieval est réutilisé.
        """

        if not answer or not context_documents:
            return answer

        answer_dates = self._extract_date_mentions(answer)

        if not answer_dates:
            return answer

        expected_range: tuple[
            date | None,
            date | None,
        ] | None = None

        if (
            parsed_query is not None
            and parsed_query.filters is not None
        ):
            expected_range = (
                parsed_query.filters.date_from,
                parsed_query.filters.date_to,
            )

        if expected_range is None:
            return answer

        answer_day_values = [
            parsed_date
            for value in answer_dates
            if (
                parsed_date := self._coerce_text_date(value)
            ) is not None
        ]

        if not answer_day_values:
            return answer

        date_from, date_to = expected_range

        if date_from is None and date_to is None:
            return answer

        if date_from is not None and date_to is not None:
            if any(
                item < date_from or item > date_to
                for item in answer_day_values
            ):
                return (
                    "Je n'ai pas trouvé d'événement correspondant "
                    "dans les informations disponibles."
                )

        if date_from is not None and date_to is None:
            if any(
                item < date_from
                for item in answer_day_values
            ):
                return (
                    "Je n'ai pas trouvé d'événement correspondant "
                    "dans les informations disponibles."
                )

        if date_from is None and date_to is not None:
            if any(
                item > date_to
                for item in answer_day_values
            ):
                return (
                    "Je n'ai pas trouvé d'événement correspondant "
                    "dans les informations disponibles."
                )

        return answer

    def __init__(self) -> None:
        """Initialise le pipeline RAG."""

        if not MISTRAL_API_KEY:
            raise RuntimeError(
                "La clé MISTRAL_API_KEY n'est pas configurée."
            )

        # --------------------------------------------------------------
        # Index
        # --------------------------------------------------------------

        self.index = EventRAGIndex()

        # --------------------------------------------------------------
        # LLM
        # --------------------------------------------------------------

        self.llm = ChatOpenAI(
            model=MISTRAL_LLM_MODEL,
            api_key=MISTRAL_API_KEY,
            base_url="https://api.mistral.ai/v1",
            temperature=0.0,
            max_tokens=2048,
        )

        # --------------------------------------------------------------
        # Composants LangChain
        # --------------------------------------------------------------

        self.query_parser: EventQueryParser | None = None
        self.retriever: EventRetriever | None = None
        self._document_chain: Any = None

        self._build_langchain_components()

    # ==================================================================
    # LANGCHAIN
    # ==================================================================

    def _build_langchain_components(self) -> None:
        """Construit les composants LangChain."""

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

La date du jour est : {current_date}

Utilise cette date de référence pour interpréter les dates
relatives présentes dans la question, comme "aujourd'hui",
"demain", "après-demain", "ce week-end", etc.

Réponds uniquement à partir des informations présentes
dans le contexte fourni.

Règles importantes :

- N'invente aucune information.
- Ne crée jamais d'événement qui n'est pas présent dans le contexte.
- Ne duplique jamais un événement déjà présent dans le contexte.
- Le nombre d'événements présentés ne doit pas dépasser
  le nombre d'événements présents dans le contexte.
- Si l'utilisateur demande plus d'événements que ceux disponibles
  dans le contexte, présente uniquement les événements réellement disponibles.
- Indique uniquement le nombre d'événements que tu présentes réellement
  dans ta réponse. Ne mentionne jamais un nombre supérieur au nombre
  d'événements effectivement listés.
- Ne crée pas et ne déduis pas d'événement supplémentaire pour atteindre
  le nombre demandé par l'utilisateur.
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

        self._document_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
            output_parser=StrOutputParser(),
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

        self._create_retriever()

    def _ensure_index_loaded(self) -> None:
        """Charge l'index depuis le disque ou le reconstruit."""

        # --------------------------------------------------------------
        # Index déjà chargé
        # --------------------------------------------------------------

        if self.index.index is not None:
            if self.retriever is None:
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
        """Crée le Retriever."""

        if self.query_parser is None:
            raise RuntimeError(
                "Le Query Parser n'est pas initialisé."
            )

        self.retriever = EventRetriever(
            index=self.index,
            query_parser=self.query_parser,
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )

        if self._document_chain is None:
            raise RuntimeError(
                "La chaîne de documents n'est pas initialisée."
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

        Pipeline :

            question
                ↓
            Query Parser
                ↓
            filtres metadata
                ↓
            FAISS
                ↓
            similarity threshold
                ↓
            TOP_K
                ↓
            documents + scores
                ├──────────────→ API
                │
                └──────────────→ documents uniquement
                                  ↓
                                LLM
                                  ↓
                                réponse
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

        if self.retriever is None:
            raise RuntimeError(
                "Le Retriever n'est pas disponible."
            )

        # --------------------------------------------------------------
        # 2. Retrieval
        # --------------------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieval_results = self.retriever.search(
            question
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # --------------------------------------------------------------
        # 3. Séparation documents / scores
        # --------------------------------------------------------------

        retrieved_documents = [
            item["document"]
            for item in retrieval_results
            if (
                isinstance(item, dict)
                and "document" in item
            )
        ]

        # Les scores restent dans cette structure pour l'API.
        # Ils ne sont jamais envoyés au LLM.
        context_results = [
            {
                "score": float(item["score"]),
                "document": item["document"],
            }
            for item in retrieval_results
            if (
                isinstance(item, dict)
                and "document" in item
                and "score" in item
            )
        ]

        print(
            f"[INFO] Retrieval                 : "
            f"{retrieval_time:.2f} s"
        )

        print(
            f"[INFO] Documents récupérés       : "
            f"{len(retrieved_documents)}"
        )

        # --------------------------------------------------------------
        # 4. Génération
        # --------------------------------------------------------------

        generation_start = time.perf_counter()

        if not retrieved_documents:
            answer_text = (
                "Je n'ai pas trouvé d'événement correspondant "
                "dans les informations disponibles."
            )
        else:
            today = datetime.now(PARIS_TZ).date()
            answer_text = self._document_chain.invoke(
                {
                    "input": question,
                    "context": retrieved_documents,
                    "current_date": today.isoformat(),
                }
            )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # --------------------------------------------------------------
        # 5. Vérification des dates
        # --------------------------------------------------------------

        parsed_query = getattr(
            self.retriever,
            "last_parsed_query",
            None,
        )

        # answer_text = self._sanitize_answer_to_context(
        #     answer_text,
        #     retrieved_documents,
        #     parsed_query,
        # )

        # --------------------------------------------------------------
        # 6. Temps total
        # --------------------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[INFO] Génération Mistral        : "
            f"{generation_time:.2f} s"
        )

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

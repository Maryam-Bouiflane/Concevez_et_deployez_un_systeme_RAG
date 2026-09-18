import os

import pandas as pd
import requests
import streamlit as st


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_URL = f"{API_BASE_URL}/ask"
EVALUATION_URL = f"{API_BASE_URL}/evaluate"


# ----------------------------------------------------------------------
# Configuration Streamlit
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="POC RAG - Évènements de la ville de Paris",
    page_icon="🎯",
    layout="wide",
)


# ----------------------------------------------------------------------
# Style global
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(
            180deg,
            #f7f9fc 0%,
            #eef4ff 100%
        );
    }

    div[data-testid="stChatMessage"] {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        color: #f8fafc;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.18);
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] span {
        color: #f8fafc !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


# ----------------------------------------------------------------------
# Sidebar - navigation
# ----------------------------------------------------------------------

with st.sidebar:

    st.title("🎯 POC RAG")
    st.caption("Système basé sur les données Open Agenda")

    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "💬 Chat Assistant",
            "📊 Évaluation du RAG",
        ],
        label_visibility="collapsed",
    )

# ======================================================================
# PAGE CHAT
# ======================================================================

if page == "💬 Chat Assistant":

    st.subheader(
        "Posez une question sur les événements de la ville de Paris"
    )

    # ------------------------------------------------------------------
    # Nouvelle conversation
    # ------------------------------------------------------------------

    if st.button(
        "Nouvelle conversation",
        key="new_conversation",
    ):
        st.session_state.messages = []
        st.session_state.last_context = []

        st.rerun()

    # ------------------------------------------------------------------
    # Historique du chat
    # ------------------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ------------------------------------------------------------------
    # Contextes récupérés
    # ------------------------------------------------------------------

    # if st.session_state.last_context:

    #     st.divider()

    #     st.subheader("🔎 Contextes récupérés")

    #     st.caption(
    #         "Documents récupérés par le retriever "
    #         "avant génération de la réponse."
    #     )

    #     for index, item in enumerate(
    #         st.session_state.last_context,
    #         start=1,
    #     ):

    #         document = item.get(
    #             "document",
    #             {},
    #         )

    #         score = item.get(
    #             "score",
    #             None,
    #         )

    #         # ----------------------------------------------------------
    #         # Document
    #         # ----------------------------------------------------------

    #         if isinstance(document, dict):

    #             metadata = document.get(
    #                 "metadata",
    #                 {},
    #             )

    #             content = document.get(
    #                 "page_content",
    #                 "",
    #             )

    #         else:

    #             metadata = {}
    #             content = str(document)

    #         title = metadata.get(
    #             "title",
    #             f"Document {index}",
    #         )

    #         # ----------------------------------------------------------
    #         # Affichage
    #         # ----------------------------------------------------------

    #         with st.expander(
    #             f"{index}. {title}"
    #         ):

    #             if score is not None:

    #                 st.write(
    #                     f"**Score de similarité :** "
    #                     f"{float(score):.3f}"
    #                 )

    #             if metadata:

    #                 city = metadata.get(
    #                     "location_city",
    #                     "N/A",
    #                 )

    #                 date_start = metadata.get(
    #                     "date_start",
    #                     "N/A",
    #                 )

    #                 st.write(
    #                     f"**Lieu :** {city}"
    #                 )

    #                 st.write(
    #                     f"**Date :** {date_start}"
    #                 )

    #             st.write(content)

    # ------------------------------------------------------------------
    # Chat input
    #
    # IMPORTANT :
    # Il est directement dans la page Chat et non dans un
    # st.tabs(). Streamlit le fixe donc au bas de la fenêtre.
    # ------------------------------------------------------------------

    prompt = st.chat_input(
        "Posez une question sur les événements..."
    )

    # ------------------------------------------------------------------
    # Traitement de la question
    # ------------------------------------------------------------------

    if prompt:

        # --------------------------------------------------------------
        # Afficher immédiatement le message utilisateur
        # --------------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # --------------------------------------------------------------
        # Appel API
        # --------------------------------------------------------------

        try:

            response = requests.post(
                API_URL,
                json={
                    "question": prompt,
                },
                timeout=120,
            )

            response.raise_for_status()

            payload = response.json()

            answer = payload.get(
                "answer",
                "Aucune réponse renvoyée.",
            )

            context_items = payload.get(
                "context",
                [],
            )

        except requests.HTTPError as exc:

            try:
                detail = response.json().get(
                    "detail",
                    "",
                )
            except Exception:
                detail = ""

            if detail:
                answer = f"Erreur API : {detail}"
            else:
                answer = f"Erreur HTTP : {exc}"

            context_items = []

        except requests.RequestException as exc:

            answer = (
                "Erreur lors de l'appel à l'API : "
                f"{exc}"
            )

            context_items = []

        except Exception as exc:

            answer = (
                "Une erreur inattendue est survenue : "
                f"{exc}"
            )

            context_items = []

        # --------------------------------------------------------------
        # Stocker et afficher la réponse
        # --------------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.last_context = context_items

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.rerun()


# ======================================================================
# PAGE ÉVALUATION
# ======================================================================

elif page == "📊 Évaluation du RAG":

    st.subheader(
        "Évaluation du système avec Ragas et Mistral."
    )

    st.markdown(
        """
        Évaluez le système RAG sur le jeu de test `data/evaluation/rag_evaluation.json`. L'évaluation peut prendre quelques minutes.
        """
    )

    # ------------------------------------------------------------------
    # Lancement de l'évaluation
    # ------------------------------------------------------------------

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        run_evaluation = st.button(
            "Lancer l'évaluation",
            type="primary",
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Appel API
    # ------------------------------------------------------------------

    if run_evaluation:

        with st.spinner(
            "Évaluation Ragas en cours..."
        ):

            try:

                response = requests.post(
                    EVALUATION_URL,
                    timeout=1800,
                )

                response.raise_for_status()

                st.session_state.evaluation_result = (
                    response.json()
                )

                st.success(
                    "Évaluation terminée."
                )

            except requests.Timeout:

                st.error(
                    "L'évaluation a dépassé le délai "
                    "d'attente."
                )

            except requests.HTTPError as exc:

                detail = ""

                try:

                    detail = response.json().get(
                        "detail",
                        "",
                    )

                except Exception:
                    pass

                if detail:

                    st.error(
                        f"Erreur API : {detail}"
                    )

                else:

                    st.error(
                        f"Erreur HTTP : {exc}"
                    )

            except requests.RequestException as exc:

                st.error(
                    "Erreur lors de l'appel à l'API : "
                    f"{exc}"
                )

            except Exception as exc:

                st.error(
                    "Une erreur inattendue est survenue : "
                    f"{exc}"
                )

    # ------------------------------------------------------------------
    # Récupération des résultats
    # ------------------------------------------------------------------

    result = st.session_state.evaluation_result

    if result:

        metrics = result.get(
            "metrics",
            {},
        )

        dataset_size = result.get(
            "dataset_size",
            0,
        )

        details = result.get(
            "details",
            [],
        )

        # --------------------------------------------------------------
        # Résumé
        # --------------------------------------------------------------

        st.divider()

        st.subheader("📋 Résumé")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.metric(
                "Questions évaluées",
                dataset_size,
            )

        with summary_col2:

            st.metric(
                "Nombre de métriques",
                len(metrics),
            )

        # --------------------------------------------------------------
        # Scores moyens
        # --------------------------------------------------------------

        st.subheader("📈 Scores moyens")

        metric_labels = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }

        metric_columns = st.columns(
            len(metrics)
        )

        for column, (
            metric_name,
            score,
        ) in zip(
            metric_columns,
            metrics.items(),
        ):

            with column:

                st.metric(
                    metric_labels.get(
                        metric_name,
                        metric_name,
                    ),
                    f"{float(score):.3f}",
                )

        # --------------------------------------------------------------
        # Graphique
        # --------------------------------------------------------------

        st.subheader(
            "📊 Comparaison des métriques"
        )

        chart_data = pd.DataFrame(
            {
                "Métrique": [
                    metric_labels.get(
                        name,
                        name,
                    )
                    for name in metrics
                ],
                "Score": [
                    float(score)
                    for score in metrics.values()
                ],
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "Métrique"
            )
        )

        # --------------------------------------------------------------
        # Résultats par question
        # --------------------------------------------------------------

        if details:

            st.divider()

            st.subheader(
                "🔍 Résultats par question"
            )

            table_data = []

            for index, item in enumerate(
                details,
                start=1,
            ):

                table_data.append(
                    {
                        "N°": index,
                        "Question": item.get(
                            "question",
                            "",
                        ),
                        "Faithfulness": item.get(
                            "faithfulness",
                            0.0,
                        ),
                        "Answer Relevancy": item.get(
                            "answer_relevancy",
                            0.0,
                        ),
                        "Context Precision": item.get(
                            "context_precision",
                            0.0,
                        ),
                        "Context Recall": item.get(
                            "context_recall",
                            0.0,
                        ),
                    }
                )

            dataframe = pd.DataFrame(
                table_data
            )

            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------------
            # Détail d'une question
            # ----------------------------------------------------------

            st.subheader(
                "🔎 Détail d'une évaluation"
            )

            selected_index = st.selectbox(
                "Sélectionnez une question",
                options=range(
                    len(details)
                ),
                format_func=lambda index: (
                    f"{index + 1}. "
                    f"{details[index].get('question', '')}"
                ),
            )

            selected = details[
                selected_index
            ]

            st.markdown(
                f"**Question :** "
                f"{selected.get('question', '')}"
            )

            st.markdown(
                "**Réponse générée :**"
            )

            st.write(
                selected.get(
                    "response",
                    "",
                )
            )

            score_col1, score_col2, score_col3, score_col4 = (
                st.columns(4)
            )

            with score_col1:

                st.metric(
                    "Faithfulness",
                    f"{float(selected.get('faithfulness', 0)):.3f}",
                )

            with score_col2:

                st.metric(
                    "Answer Relevancy",
                    f"{float(selected.get('answer_relevancy', 0)):.3f}",
                )

            with score_col3:

                st.metric(
                    "Context Precision",
                    f"{float(selected.get('context_precision', 0)):.3f}",
                )

            with score_col4:

                st.metric(
                    "Context Recall",
                    f"{float(selected.get('context_recall', 0)):.3f}",
                )

    else:

        st.info(
            "Aucune évaluation disponible. "
            "Cliquez sur « Lancer l'évaluation » pour commencer."
        )
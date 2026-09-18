import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL.rstrip('/')}/ask"

st.set_page_config(
    page_title="POC RAG Open Agenda",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef4ff 100%);
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
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 POC RAG Open Agenda")
st.caption("Interface de démonstration du système RAG basé sur les événements Open Agenda (Le backend FastAPI récupère les événements, filtre, récupère les contextes et génère la réponse).")

if st.button("Nouvelle conversation"):
    st.session_state.messages = []
    st.session_state.last_context = []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_context" not in st.session_state:
    st.session_state.last_context = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Posez une question sur les événements...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            API_URL,
            json={"question": prompt},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer", "Aucune réponse renvoyée.")
        context_items = payload.get("context", [])
    except Exception as exc:
        answer = f"Erreur lors de l'appel à l'API : {exc}"
        context_items = []

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_context = context_items

    with st.chat_message("assistant"):
        st.markdown(answer)


"""Construit l'index FAISS à partir des événements OpenAgenda.

Usage:
    uv run python scripts/build_index.py

Le script :
    1. récupère les événements OpenAgenda ;
    2. prépare les documents ;
    3. génère les embeddings avec mistral-embed ;
    4. construit l'index FAISS ;
    5. sauvegarde l'index et les métadonnées dans data/.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Permet d'exécuter le script depuis la racine du projet avec :
#
#     uv run python scripts/build_index.py
#
# tout en permettant à Python de trouver le package "app".

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.indexer import EventRAGIndex  # noqa: E402


def main() -> None:
    """Construit et sauvegarde l'index FAISS."""

    print("=" * 70)
    print("CONSTRUCTION DE L'INDEX RAG")
    print("=" * 70)

    print("\nInitialisation de EventRAGIndex...")

    index = EventRAGIndex()

    print("Construction de l'index en cours...")
    print(
        "Cette étape peut prendre du temps car les embeddings "
        "sont générés via l'API Mistral."
    )

    print()

    index.build_index()

    print("\n" + "=" * 70)
    print("INDEX CONSTRUIT AVEC SUCCÈS")
    print("=" * 70)

    print("\nFichiers générés :")
    print("  - data/faiss_index.bin")
    print("  - data/metadata.json")

    print("\nLe système RAG est maintenant prêt à être utilisé.")


if __name__ == "__main__":
    main()

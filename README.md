# POC RAG Open Agenda

Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** permettant de répondre à des questions en langage naturel à partir d'événements provenant de l'API publique **Open Agenda**.

Le système combine :

* l'API Open Agenda pour récupérer les événements ;
* les embeddings **Mistral** pour représenter les documents et les requêtes ;
* **FAISS** pour effectuer la recherche sémantique ;
* un seuil de similarité pour filtrer les résultats peu pertinents ;
* **Mistral** pour générer une réponse à partir des événements récupérés ;
* **Ragas** pour évaluer automatiquement la qualité du système RAG ;
* **FastAPI** pour exposer le système sous forme d'API REST ;
* **pytest** pour tester les différents composants du projet.

---

## Architecture

Le fonctionnement global du système est le suivant :

```text
                    ┌──────────────────────┐
                    │     Open Agenda      │
                    │     API publique     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Data Loader       │
                    │ Normalisation /      │
                    │ déduplication        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Indexer         │
                    │ Documents +          │
                    │ embeddings Mistral   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FAISS          │
                    │ Index vectoriel      │
                    └──────────┬───────────┘
                               │
                 question      │
                    │          │
                    ▼          │
              ┌──────────┐     │
              │ Embedding│     │
              │ Mistral  │     │
              └────┬─────┘     │
                   │           │
                   └─────┬─────┘
                         ▼
                  Recherche FAISS
                         │
                         ▼
              Filtrage par similarité
                         │
                         ▼
                Contextes pertinents
                         │
                         ▼
                 ┌──────────────┐
                 │    Mistral   │
                 │ LLM génératif│
                 └──────┬───────┘
                        │
                        ▼
                    Réponse
```

---

## Structure du projet

```text
.
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── indexer.py
│   │   └── rag_service.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset.py
│       └── ragas_evaluation.py
│
├── data/
│   ├── evaluation/
│   │   └── rag_evaluation.json
│   ├── events_sample.json
│   ├── faiss_index.bin
│   └── metadata.json
│
├── tests/
│   ├── test_api.py
│   ├── test_data_loader.py
│   ├── test_dataset.py
│   ├── test_indexer.py
│   ├── test_rag_service.py
│   └── test_ragas_evaluation.py
│
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```

### Rôle des principaux modules

| Module                               | Rôle                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| `app/api.py`                         | Expose le système RAG via FastAPI                            |
| `app/config.py`                      | Centralise la configuration et les variables d'environnement |
| `app/core/data_loader.py`            | Récupère et normalise les événements Open Agenda             |
| `app/core/indexer.py`                | Construit les documents, embeddings et index FAISS           |
| `app/core/rag_service.py`            | Orchestre la recherche et la génération de réponse           |
| `app/evaluation/dataset.py`          | Charge le dataset de référence pour l'évaluation             |
| `app/evaluation/ragas_evaluation.py` | Construit le dataset Ragas et exécute l'évaluation           |
| `tests/`                             | Tests unitaires et fonctionnels du projet                    |

---

## Prérequis

* **Python 3.12**
* **uv**
* Une clé API **Mistral** pour utiliser les embeddings et la génération de réponses.

---

## Installation

Clonez le projet puis installez les dépendances avec `uv` :

```bash
uv sync
```

Le projet utilise les dépendances déclarées dans `pyproject.toml`.

---

## Configuration

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=your_key_here

SIMILARITY_THRESHOLD=0.45
TOP_K=3
DEFAULT_CITY=Paris
LOOKBACK_DAYS=365
```

### Variables disponibles

| Variable               | Description                                        | Valeur par défaut |
| ---------------------- | -------------------------------------------------- | ----------------- |
| `MISTRAL_API_KEY`      | Clé API Mistral                                    | aucune            |
| `SIMILARITY_THRESHOLD` | Seuil minimal de similarité FAISS                  | `0.45`            |
| `TOP_K`                | Nombre maximal de documents récupérés              | `3`               |
| `DEFAULT_CITY`         | Ville utilisée par défaut dans les requêtes        | `Paris`           |
| `LOOKBACK_DAYS`        | Nombre de jours pris en compte pour les événements | `365`             |

Le fichier `.env` ne doit pas être commité dans le dépôt.

---

## Lancer l'API

Lancez le serveur FastAPI avec :

```bash
uv run uvicorn app.api:app --reload
```

L'API sera accessible à l'adresse :

```text
http://127.0.0.1:8000
```

La documentation interactive Swagger est disponible sur :

```text
http://127.0.0.1:8000/docs
```

---

## Endpoint principal

### `POST /ask`

Permet de poser une question au système RAG.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quels événements à Paris ?\"}"
```

Le système :

1. transforme la question en embedding ;
2. recherche les documents les plus proches dans FAISS ;
3. applique le seuil de similarité ;
4. transmet les contextes pertinents au LLM Mistral ;
5. génère une réponse à partir des événements récupérés.

---

## Reconstituer l'index

L'API permet également de reconstruire l'index à partir des données disponibles.

L'endpoint correspondant est :

```text
POST /rebuild-index
```

Il permet de relancer le processus d'indexation :

```text
Open Agenda
     ↓
normalisation des événements
     ↓
création des documents
     ↓
embeddings Mistral
     ↓
index FAISS
```

Les fichiers générés sont notamment :

```text
data/faiss_index.bin
data/metadata.json
```

---

## Fonctionnement du RAG

```
Question complète utilisateur
          ↓
     Query Parser
          ↓
    filtres metadata
          ↓
  documents candidats
          ↓
       FAISS
          ↓
     threshold
          ↓
       top_k
          ↓
     Documents
          ↓
        LLM
```

Le système utilise une architecture RAG en deux étapes.

### 1. Retrieval

Chaque événement est transformé en document puis en vecteur grâce au modèle d'embedding :

```text
mistral-embed
```

Les vecteurs sont stockés dans un index **FAISS**.

Lorsqu'un utilisateur pose une question, celle-ci est également transformée en embedding puis comparée aux vecteurs des événements.

Le système récupère les `TOP_K` documents les plus proches et élimine ceux dont la similarité est inférieure à :

```env
SIMILARITY_THRESHOLD=0.45
```

### 2. Generation

Les documents récupérés sont ensuite transmis au modèle :

```text
mistral-small-latest
```

Le modèle génère une réponse à partir des contextes fournis par le retrieval.

Si aucun document suffisamment pertinent n'est trouvé, le système retourne une réponse de secours au lieu de générer une réponse potentiellement non fondée.

---

## Évaluation avec Ragas

Le projet contient également une évaluation automatique de la qualité du système RAG avec **Ragas**.

Le dataset d'évaluation se trouve dans :

```text
data/evaluation/rag_evaluation.json
```

Chaque exemple contient notamment :

```json
{
  "user_input": "Quels événements sont proposés à Paris ?",
  "reference": "..."
}
```

Le dataset contient uniquement les questions et leurs réponses de référence.

Les réponses et les contextes ne sont **pas écrits manuellement dans ce fichier** : ils sont produits par le véritable système RAG lors de l'évaluation.

Le processus est donc :

```text
rag_evaluation.json
        │
        ▼
_build_ragas_dataset()
        │
        ▼
EventRAGService.answer()
        │
        ├── response
        └── retrieved_contexts
        │
        ▼
EvaluationDataset Ragas
        │
        ▼
Ragas
        │
        ├── Faithfulness
        ├── AnswerRelevancy
        ├── ContextPrecision
        └── ContextRecall
```

### Lancer l'évaluation

L'évaluation Ragas utilise les modèles Mistral comme modèles d'évaluation et nécessite donc une clé API :

```env
MISTRAL_API_KEY=your_key_here
```

La fonction principale est :

```python
from app.evaluation.ragas_evaluation import run_ragas_evaluation

result = run_ragas_evaluation()
```

L'évaluation produit les scores des différentes métriques Ragas.

> Les tests unitaires de `ragas_evaluation.py` utilisent des mocks afin de ne pas effectuer d'appels réels à l'API Mistral pendant l'exécution de pytest.

---

## Tests

Les tests couvrent les principaux composants du projet :

* API FastAPI ;
* chargement des événements ;
* normalisation et déduplication ;
* chargement du dataset d'évaluation ;
* création des documents ;
* génération des embeddings ;
* construction de l'index FAISS ;
* recherche avec seuil de similarité ;
* service RAG ;
* construction du dataset Ragas ;
* orchestration de l'évaluation Ragas.

Pour lancer l'ensemble des tests :

```bash
uv run pytest -vv
```

Les tests doivent être exécutés sans nécessiter de véritables appels à l'API Mistral : les appels externes sont mockés lorsque cela est nécessaire.

---

## Couverture des tests

Pour générer le rapport de couverture :

```bash
uv run pytest --cov=app --cov-report=term-missing
```

Le rapport indique, pour chaque module, les lignes couvertes ou non couvertes par les tests.

---

## Dépendances principales

| Technologie | Utilisation                                   |
| ----------- | --------------------------------------------- |
| Python 3.12 | Langage                                       |
| FastAPI     | API REST                                      |
| Uvicorn     | Serveur ASGI                                  |
| Open Agenda | Source des événements                         |
| Mistral AI  | Embeddings et génération de texte             |
| FAISS       | Recherche vectorielle                         |
| LangChain   | Gestion des documents et intégration RAG      |
| Ragas       | Évaluation du système RAG                     |
| pytest      | Tests                                         |
| uv          | Gestion de l'environnement et des dépendances |

---

## Limites du POC

Ce projet constitue un **POC (Proof of Concept)** et présente donc plusieurs limites :

* les données Open Agenda sont récupérées depuis une API externe ;
* l'index FAISS est stocké localement ;
* les embeddings et la génération Mistral nécessitent une connexion à l'API Mistral ;
* aucune base vectorielle persistante distribuée n'est utilisée ;
* le système n'intègre pas encore de mécanisme avancé de reranking ;
* les performances dépendent de la qualité des événements disponibles dans Open Agenda et de leur représentation vectorielle.

L'objectif principal est de démontrer le fonctionnement de bout en bout d'une architecture RAG simple, testable et évaluable.

---

## Résumé

Ce projet met en œuvre une chaîne RAG complète :

```text
Open Agenda
    ↓
Data Loader
    ↓
Documents
    ↓
Mistral Embeddings
    ↓
FAISS
    ↓
Recherche sémantique
    ↓
Filtrage par similarité
    ↓
Contextes pertinents
    ↓
Mistral LLM
    ↓
Réponse utilisateur
    ↓
Ragas
    ↓
Évaluation de la qualité du RAG
```

L'architecture est volontairement simple afin de permettre de comprendre et de tester chaque étape indépendamment.

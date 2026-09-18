# POC RAG Open Agenda

Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** permettant de répondre à des questions en langage naturel à partir d'événements provenant de l'API publique **Open Agenda**.

Le système combine :

* **Open Agenda** pour récupérer les événements ;
* **Mistral Embeddings** (`mistral-embed`) pour transformer les événements et les questions en vecteurs ;
* **FAISS** pour effectuer la recherche vectorielle ;
* un **Query Parser** basé sur un LLM Mistral et une sortie structurée Pydantic pour extraire les contraintes temporelles et géographiques ;
* un **pré-filtrage par métadonnées** avant la recherche vectorielle ;
* un **seuil de similarité** pour éliminer les résultats trop peu pertinents ;
* **Mistral Small** (`mistral-small-latest`) pour générer la réponse finale ;
* **LangChain** pour orchestrer les documents, le retriever et la chaîne RAG ;
* **Ragas** pour évaluer automatiquement la qualité du système RAG ;
* **FastAPI** pour exposer le système sous forme d'API REST ;
* **pytest** pour tester les différents composants du projet.

L'objectif de ce POC est de mettre en œuvre une architecture RAG complète, simple, testable et compréhensible, depuis la récupération des données jusqu'à la génération et l'évaluation des réponses.

---

# Démarrage rapide

## 1. Installer les dépendances

```bash
uv sync
```

## 2. Configurer les variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=your_key_here
SIMILARITY_THRESHOLD=0.45
TOP_K=3
DEFAULT_CITY=Paris
LOOKBACK_DAYS=365
```

## 3. Lancer l'API localement

```bash
uv run uvicorn app.api:app --reload
```

L'API est alors accessible sur :

```text
http://127.0.0.1:8000
```

Swagger est disponible sur :

```text
http://127.0.0.1:8000/docs
```

## 4. Lancer l'interface Streamlit localement

Le projet dispose aussi d'une interface utilisateur légère en Streamlit, qui appelle l'API backend sans contenir la logique RAG.

```bash
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run streamlit_app.py --server.port 8501
```

L'interface est alors accessible sur :

```text
http://127.0.0.1:8501
```

## 5. Lancer avec Docker

### Option A : backend + UI dans un seul lancement avec Docker Compose

```bash
docker compose up --build
```

Cela démarre :

- le backend FastAPI sur `http://localhost:8000`
- l'interface Streamlit sur `http://localhost:8501`

### Option B : lancer seulement le backend API

```bash
# 1) construit l'image Docker du projet
docker build -t rag-app .

# 2) supprime un ancien conteneur du même nom si besoin
docker rm -f rag-api 2>/dev/null || true

# 3) démarre le conteneur en arrière-plan avec les variables du .env
docker run -d --env-file .env -p 8000:8000 --name rag-api rag-app
```

### Option C : lancer seulement l'interface Streamlit

```bash
docker build -f Dockerfile.streamlit -t rag-ui .
docker rm -f rag-ui 2>/dev/null || true
docker run -d -p 8501:8501 --name rag-ui -e API_BASE_URL=http://host.docker.internal:8000 rag-ui
```

Le conteneur démarre automatiquement l'interface Streamlit sur le port `8501`.

## 6. Vérifier rapidement

Ouvrez dans le navigateur :

```text
http://localhost:8000/
```

Pour consulter la documentation Swagger :

```text
http://localhost:8000/docs
```

Pour ouvrir l'interface utilisateur :

```text
http://localhost:8501/
```

Ou, en ligne de commande :

```bash
curl http://localhost:8000/
```

---

## Architecture générale

Le fonctionnement global du projet est le suivant :

```text
                         ┌────────────────────────┐
                         │      Open Agenda       │
                         │       API publique     │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │      Data Loader       │
                         │                        │
                         │ Récupération           │
                         │ Normalisation          │
                         │ Déduplication          │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │        Indexer         │
                         │                        │
                         │ Documents LangChain    │
                         │ Embeddings Mistral     │
                         │ Métadonnées            │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │         FAISS          │
                         │                        │
                         │ Index vectoriel        │
                         └────────────────────────┘


                    QUESTION UTILISATEUR
                              │
                              ▼
                   ┌──────────────────────┐
                   │    Query Parser      │
                   │                      │
                   │ LLM Mistral          │
                   │ + Pydantic           │
                   └──────────┬───────────┘
                              │
                              ▼
                       Filtres metadata
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Pré-filtrage         │
                   │ des documents        │
                   └──────────┬───────────┘
                              │
                              ▼
                Documents candidats uniquement
                              │
                    QUESTION UTILISATEUR
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Mistral Embeddings   │
                   │    mistral-embed     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │        FAISS         │
                   │ Recherche vectorielle│
                   └──────────┬───────────┘
                              │
                              ▼
                    Seuil de similarité
                              │
                              ▼
                           TOP_K
                              │
                              ▼
                   Documents pertinents
                              │
                              ▼
                   ┌──────────────────────┐
                   │       Mistral        │
                   │ mistral-small-latest │
                   └──────────┬───────────┘
                              │
                              ▼
                         Réponse finale
```

---

# Structure du projet

```text
.
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
│
├── app/
│   ├── api.py
│   ├── config.py
│   │
│   ├── core/
│   │   ├── data_loader.py
│   │   ├── indexer.py
│   │   ├── query_parser.py
│   │   ├── rag_service.py
│   │   └── retriever.py
│   │
│   ├── evaluation/
│   │   ├── dataset.py
│   │   └── ragas_evaluation.py
│   │
│   ├── schemas/
│   │   └── search.py
│   │
│   └── scripts/
│       ├── build_index.py
│       └── resolve_reference_contexts.py
│
├── data/
│   ├── faiss_index.bin
│   ├── metadata.json
│   └── evaluation/
│       ├── rag_evaluation.json
│       └── ragas_dataset.json
│
└── tests/
    ├── test_api.py
    ├── test_data_loader.py
    ├── test_dataset.py
    ├── test_indexer.py
    ├── test_mistral.py
    ├── test_query_parser.py
    ├── test_ragas_evaluation.py
    ├── test_ragas_requires_reference_contexts.py
    ├── test_rag_service.py
    ├── test_reference_contexts_script.py
    └── test_retriever.py
```

## Rôle des principaux modules

| Module                                                    | Rôle                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| `main.py`                                                 | Point d'entrée principal du projet et lancement local de l'API      |
| `Dockerfile`                                              | Construit l'image et démarre automatiquement l'API dans le conteneur |
| `app/api.py`                                              | Expose le système RAG via FastAPI                                   |
| `app/config.py`                                           | Centralise la configuration et les variables d'environnement        |
| `app/core/data_loader.py`                                 | Récupère, normalise et prépare les événements Open Agenda           |
| `app/core/indexer.py`                                     | Crée les documents, génère les embeddings et construit l'index FAISS |
| `app/core/query_parser.py`                                | Extrait les contraintes temporelles et géographiques de la question |
| `app/core/retriever.py`                                   | Applique les filtres metadata puis effectue la recherche vectorielle |
| `app/core/rag_service.py`                                 | Orchestre le retrieval, la génération et la validation de la réponse |
| `app/schemas/search.py`                                   | Définit les modèles Pydantic utilisés par le Query Parser            |
| `app/evaluation/dataset.py`                               | Charge le dataset de référence et les jeux d'évaluation             |
| `app/evaluation/ragas_evaluation.py`                      | Construit le dataset Ragas et exécute l'évaluation                 |
| `app/scripts/build_index.py`                              | Script pour reconstruire l'index vectoriel                         |
| `app/scripts/resolve_reference_contexts.py`               | Script pour compléter et valider les `reference_contexts`         |
| `data/`                                                   | Stocke le dataset d'évaluation, les métadonnées et l'index FAISS   |
| `tests/test_ragas_requires_reference_contexts.py`         | Vérifie que l'évaluation exige bien les contextes de référence     |
| `tests/test_reference_contexts_script.py`                 | Vérifie la logique de résolution des `reference_contexts`          |
| `tests/`                                                  | Contient les tests des différents composants                        |

---

# Prérequis

* **Python 3.12**
* **uv**
* Une clé API **Mistral AI**
* Une connexion Internet pour :

  * récupérer les événements Open Agenda ;
  * utiliser les embeddings Mistral ;
  * utiliser le LLM Mistral ;
  * effectuer l'évaluation Ragas.

---

# Installation

Clonez le projet puis installez les dépendances avec `uv` :

```bash
uv sync
```

Les dépendances utilisées par le projet sont définies dans `pyproject.toml`.

Pour vérifier la version de Python utilisée :

```bash
uv run python --version
```

Pour vérifier la version de `uv` :

```bash
uv --version
```

---

# Configuration

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=your_key_here

SIMILARITY_THRESHOLD=0.45
TOP_K=3

DEFAULT_CITY=Paris
LOOKBACK_DAYS=365
```

## Variables d'environnement

| Variable               | Description                                                          | Valeur par défaut |
| ---------------------- | -------------------------------------------------------------------- | ----------------- |
| `MISTRAL_API_KEY`      | Clé API Mistral utilisée pour les embeddings, le LLM et l'évaluation | Aucune            |
| `SIMILARITY_THRESHOLD` | Seuil minimal de similarité pour conserver un document               | `0.45`            |
| `TOP_K`                | Nombre maximal de documents récupérés                                | `3`               |
| `DEFAULT_CITY`         | Ville utilisée lors de la récupération des événements                | `Paris`           |
| `LOOKBACK_DAYS`        | Période de récupération des événements                               | `365`             |

Le fichier `.env` contient une clé secrète et **ne doit pas être commité dans Git**.

Il doit être ajouté au `.gitignore` :

```gitignore
.env
```

---

# Lancement avec Docker

Le projet contient un `Dockerfile` qui construit l'image Python du service et démarre automatiquement l'API FastAPI au lancement du conteneur.

Construire l'image :

```bash
docker build -t rag-app .
```

Lancer le conteneur sur le port `8000` :

```bash
docker run -d -p 8000:8000 --name rag-api rag-app
```

Le conteneur démarre automatiquement avec la commande suivante définie dans le `Dockerfile` :

```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Vous pouvez ensuite vérifier que l'API répond :

```bash
curl http://localhost:8000/
```

---

# Acquisition et préparation des données

Les événements sont récupérés depuis l'API publique Open Agenda.

Le `Data Loader` est responsable de :

1. récupérer les événements ;
2. normaliser les différents champs ;
3. gérer les valeurs manquantes ;
4. extraire les informations temporelles ;
5. extraire les informations géographiques ;
6. construire le contenu textuel utilisé pour les embeddings ;
7. effectuer la déduplication lorsque nécessaire.

Les informations exploitées comprennent notamment :

* identifiant de l'événement ;
* titre ;
* description ;
* description longue ;
* mots-clés ;
* conditions ;
* dates de début et de fin ;
* nom du lieu ;
* adresse ;
* ville ;
* arrondissement/quartier ;
* code postal ;
* département ;
* région ;
* pays ;
* coordonnées géographiques.

Le contenu textuel de chaque événement est utilisé pour générer son embedding.

---

# Construction de l'index

L'indexation est réalisée par `app/core/indexer.py`.

Pour chaque événement, le système construit un objet `Document` LangChain contenant :

```text
Document
├── page_content
│   └── contenu textuel de l'événement
│
└── metadata
    ├── uid
    ├── title
    ├── date_start
    ├── date_end
    ├── location_city
    ├── location_district
    ├── location_postalcode
    ├── location_department
    ├── location_region
    ├── location_countrycode
    └── country
```

Le `page_content` est transformé en vecteur avec le modèle :

```text
mistral-embed
```

Les vecteurs sont ensuite stockés dans un index :

```text
FAISS IndexFlatIP
```

`IndexFlatIP` utilise le produit scalaire (`Inner Product`) pour comparer les vecteurs.

Les embeddings utilisés dans ce projet étant normalisés, le produit scalaire permet d'obtenir une mesure proche de la **similarité cosinus**.

Les fichiers générés sont notamment :

```text
data/faiss_index.bin
data/metadata.json
```

* `faiss_index.bin` contient l'index vectoriel ;
* `metadata.json` contient les documents et leurs métadonnées associées.

---

# Fonctionnement du Query Parser

Le système utilise un **Query Parser** avant la recherche vectorielle.

Son rôle est de transformer la question utilisateur en contraintes pouvant être appliquées aux métadonnées.

Le Query Parser utilise :

* un LLM Mistral ;
* `ChatPromptTemplate` ;
* une sortie structurée Pydantic ;
* le schéma `EventSearchQuery`.

Le parser extrait uniquement :

1. les contraintes temporelles ;
2. les contraintes géographiques.

Il **ne réécrit pas la question sémantique**.

Par exemple :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

peut être transformé en :

```text
filters:
    date_from: 2026-09-05
    date_to: 2026-09-06
    location_city: Paris
```

La partie :

```text
concerts de jazz
pour les familles
```

reste dans la question originale et sera utilisée pour la recherche vectorielle.

## Contraintes temporelles

Le Query Parser est capable d'interpréter notamment :

* aujourd'hui ;
* demain ;
* après-demain ;
* dans plusieurs jours ;
* ce week-end ;
* le week-end prochain ;
* cette semaine ;
* la semaine prochaine ;
* la semaine dernière ;
* dans deux semaines ;
* ce mois-ci ;
* le mois prochain ;
* le mois dernier ;
* cette année ;
* l'année prochaine ;
* l'année dernière ;
* une date précise ;
* une période précise ;
* un mois précis ;
* une année précise.

Les expressions relatives sont converties en dates réelles à partir de la date actuelle.

Par exemple :

```text
demain
```

devient :

```text
date_from = date de demain
date_to   = date de demain
```

Pour une période :

```text
date_from = début de la période
date_to   = fin de la période
```

## Contraintes géographiques

Le parser peut extraire notamment :

* `location_city`
* `location_district`
* `location_postalcode`
* `location_department`
* `location_region`
* `location_countrycode`
* `country_fr`

Une règle importante est appliquée :

> Une information géographique n'est renseignée que si elle apparaît explicitement dans la question utilisateur.

Le système ne déduit donc pas automatiquement :

```text
Paris → France
Paris → Île-de-France
Paris → département 75
```

Par exemple :

```text
Quels événements ont lieu à Paris ?
```

produit uniquement :

```text
location_city = Paris
```

Les autres champs géographiques restent `null`.

---

# Fonctionnement du Retrieval

Le retrieval utilise une architecture en deux étapes :

```text
Question utilisateur
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
 Query Parser                  Question originale
        │                             │
        ▼                             │
Filtres metadata                     │
        │                             │
        ▼                             │
Documents candidats                  │
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
             Embedding de la
             question complète
                       │
                       ▼
                    FAISS
                       │
                       ▼
             Similarity threshold
                       │
                       ▼
                    TOP_K
                       │
                       ▼
              Documents pertinents
```

## 1. Extraction des filtres

La question est d'abord envoyée au Query Parser.

Par exemple :

```text
Quels concerts sont prévus ce week-end à Paris ?
```

peut produire :

```text
date_from = 2026-09-05
date_to = 2026-09-06
location_city = Paris
```

## 2. Pré-filtrage metadata

Les métadonnées sont ensuite utilisées pour sélectionner les documents candidats.

Le filtrage est effectué **avant la recherche FAISS**.

Par exemple :

```text
Tous les événements
       ↓
Ville = Paris
       ↓
Date comprise dans la période demandée
       ↓
Documents candidats
```

Les filtres sont combinés avec une logique `AND`.

Ainsi, un événement doit respecter l'ensemble des contraintes explicites.

### Filtrage temporel

Un événement est conservé lorsqu'il chevauche la période recherchée.

Cela permet notamment de conserver un événement qui commence avant la période demandée mais se poursuit pendant celle-ci.

## 3. Recherche vectorielle

Une fois les documents candidats déterminés, la **question complète de l'utilisateur** est transformée en embedding avec :

```text
mistral-embed
```

Cette question complète est importante car elle contient les informations sémantiques nécessaires à la recherche.

Par exemple :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

n'est pas réduite à :

```text
concerts
```

ou :

```text
jazz
```

L'embedding est calculé à partir de la question complète.

## 4. Recherche FAISS

FAISS compare l'embedding de la question avec les embeddings des documents candidats.

Si aucun filtre metadata n'est présent, la recherche est effectuée directement dans l'index FAISS global.

Si des filtres sont présents, seuls les documents correspondants sont utilisés pour construire une recherche FAISS temporaire.

Cela permet de réaliser :

```text
Filtres metadata
        ↓
Documents candidats
        ↓
Recherche vectorielle
```

plutôt que :

```text
Recherche vectorielle globale
        ↓
Filtrage metadata
```

Cette distinction permet de respecter strictement les contraintes explicites de l'utilisateur.

## 5. Seuil de similarité

Les résultats dont le score de similarité est inférieur à :

```env
SIMILARITY_THRESHOLD=0.45
```

sont supprimés.

Cela évite de transmettre au LLM des documents trop éloignés de la question.

## 6. TOP_K

Le paramètre :

```env
TOP_K=3
```

limite le nombre maximal de documents transmis au système de génération.

Par exemple :

```text
100 documents candidats
        ↓
Recherche FAISS
        ↓
TOP_K = 3
        ↓
3 documents maximum
```

Le nombre réel de documents peut être inférieur à `TOP_K` si :

* moins de documents candidats existent ;
* certains documents sont sous le seuil de similarité.

## 7. Aucun fallback sur les filtres explicites

Lorsqu'un utilisateur fournit un filtre explicite, le système **ne revient pas à une recherche globale** si aucun document ne correspond.

Par exemple :

```text
Quels événements sont prévus à Bordeaux ?
```

Si aucun événement ne correspond à `location_city = Bordeaux`, le système retourne zéro document.

Il ne recherche pas automatiquement dans tous les événements.

Cela évite de retourner des événements qui ne respectent pas la contrainte demandée.

---

# Génération de la réponse

Après le retrieval, les documents pertinents sont transmis au LLM :

```text
mistral-small-latest
```

Le LLM reçoit la question ainsi que les documents récupérés comme contexte.

Le principe est :

```text
Question utilisateur
        +
Documents récupérés
        ↓
Mistral Small
        ↓
Réponse
```

Le modèle doit donc générer sa réponse à partir des informations récupérées par le système RAG.

L'objectif est de limiter les réponses non fondées en fournissant au modèle un contexte provenant des événements Open Agenda.

Si aucun document suffisamment pertinent n'est trouvé, aucune information documentaire pertinente n'est fournie au LLM pour répondre à la question.

---

# Architecture LangChain

LangChain est utilisé principalement pour structurer le pipeline RAG.

Le retriever implémente l'interface `BaseRetriever`.

Le flux est donc conceptuellement :

```text
EventRAGService
       │
       ▼
EventRetriever
       │
       ├── Query Parser
       │
       ├── Metadata filtering
       │
       ├── Mistral embedding
       │
       ├── FAISS
       │
       ├── Similarity threshold
       │
       └── TOP_K
       │
       ▼
List[Document]
       │
       ▼
Document chain
       │
       ▼
Mistral LLM
       │
       ▼
Réponse
```

La séparation des responsabilités permet de tester indépendamment :

* le parsing de la question ;
* le filtrage metadata ;
* la recherche vectorielle ;
* le service RAG ;
* la génération ;
* l'évaluation.

---

# API FastAPI

L'application est exposée via FastAPI.

## Lancer l'API

Depuis la racine du projet :

```bash
uv run uvicorn app.api:app --reload
```

L'API est alors disponible à :

```text
http://127.0.0.1:8000
```

La documentation Swagger interactive est disponible à :

```text
http://127.0.0.1:8000/docs
```

La documentation ReDoc est disponible à :

```text
http://127.0.0.1:8000/redoc
```

---

# Endpoints principaux

## `POST /ask`

Cet endpoint permet de poser une question au système RAG.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quels concerts de jazz sont prévus ce week-end à Paris ?\"}"
```

Le traitement est alors :

```text
POST /ask

    │
    ▼
Question utilisateur

    │
    ▼
Query Parser

    │
    ▼
Filtres metadata

    │
    ▼
Pré-filtrage

    │
    ▼
Embedding de la question complète

    │
    ▼
FAISS

    │
    ▼
Threshold

    │
    ▼
TOP_K

    │
    ▼
Documents

    │
    ▼
Mistral Small

    │
    ▼
Réponse
```

## `POST /rebuild`

Cet endpoint permet de **reconstruire l'index du système RAG**.

Il récupère les événements depuis l'API OpenAgenda, prépare les documents, génère leurs embeddings avec `mistral-embed`, puis reconstruit et sauvegarde l'index vectoriel FAISS ainsi que les métadonnées associées.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/rebuild
```

Le traitement est alors :

```text
POST /rebuild

    │
    ▼
API OpenAgenda

    │
    ▼
Récupération des événements

    │
    ▼
Préparation / normalisation des données

    │
    ▼
Création des Documents

    │
    ▼
mistral-embed

    │
    ▼
Génération des embeddings

    │
    ▼
Construction de l'index FAISS

    │
    ▼
Sauvegarde de l'index
et des métadonnées

    │
    ▼
Index RAG mis à jour
```

> **Attention :** la reconstruction de l'index nécessite de recalculer les embeddings des événements et peut donc entraîner de nombreux appels à l'API Mistral.

## `POST /evaluate`

L'endpoint `/evaluate` lance l'évaluation du système RAG avec **Ragas**.

À chaque appel, il :

1. charge le dataset de référence depuis [data/evaluation/rag_evaluation.json](data/evaluation/rag_evaluation.json),
2. exécute le pipeline RAG complet sur chacune des questions du dataset,
3. récupère la réponse générée ainsi que les contextes retrouvés,
4. met à jour ces valeurs dans le JSON de référence,
5. calcule les métriques d'évaluation.

Le jeu de données contient pour chaque exemple :

* une question (`user_input`),
* une référence attendue (`reference`),
* les contextes de référence (`reference_contexts`),
* la réponse calculée par le RAG (`response`),
* les contextes récupérés (`retrieved_contexts`).

Les métriques calculées sont notamment :

* **Faithfulness**
* **Answer Relevancy**
* **Context Precision**
* **Context Recall**

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/evaluate
```

---

# Reconstruction de l'index

L'index peut être reconstruit avec l'endpoint :

```text
POST /rebuild
```

La reconstruction relance le pipeline d'indexation :

```text
Open Agenda
     ↓
Récupération des événements
     ↓
Normalisation
     ↓
Déduplication
     ↓
Création des Documents
     ↓
Mistral Embeddings
     ↓
FAISS
     ↓
Sauvegarde
```

Les principaux fichiers produits sont :

```text
data/faiss_index.bin
data/metadata.json
```

## Quand faut-il reconstruire l'index ?

Il est nécessaire de reconstruire l'index lorsque les données ou leur représentation changent.

Par exemple :

* nouveaux événements Open Agenda ;
* modification de la préparation du `page_content` ;
* modification des métadonnées indexées ;
* changement du modèle d'embedding ;
* modification de la méthode de génération des embeddings.

En revanche, les modifications suivantes ne nécessitent généralement pas de reconstruire l'index :

* modification du Query Parser ;
* modification du retriever ;
* modification du threshold ;
* modification de `TOP_K` ;
* modification de la logique du service RAG ;
* modification des prompts de génération ;
* modification des tests.

---

# Évaluation avec Ragas

Le projet contient une évaluation automatique du système RAG avec **Ragas**.

Le dataset de référence se trouve dans :

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

Le fichier contient les questions et les réponses de référence.

Les réponses générées et les contextes récupérés ne sont pas écrits manuellement dans le dataset.

Ils sont produits par le véritable système RAG pendant l'évaluation.

---

# Pipeline d'évaluation

Le processus est le suivant :

```text
rag_evaluation.json
        │
        ▼
Chargement du dataset
        │
        ▼
EventRAGService.answer()
        │
        ├── Réponse générée
        │
        └── Documents récupérés
        │
        ▼
Construction du EvaluationDataset
        │
        ▼
Ragas
        │
        ├── Faithfulness
        ├── Answer Relevancy
        ├── Context Precision
        └── Context Recall
        │
        ▼
Scores d'évaluation
```

## Métriques utilisées

### Faithfulness

Mesure dans quelle mesure la réponse générée est supportée par le contexte récupéré.

### Answer Relevancy

Mesure la pertinence de la réponse par rapport à la question utilisateur.

### Context Precision

Évalue la pertinence des documents récupérés par rapport à la question.

### Context Recall

Évalue si les informations nécessaires à la réponse sont présentes dans les contextes récupérés.

---

# Lancer l'évaluation Ragas

L'évaluation utilise les modèles Mistral et nécessite donc :

```env
MISTRAL_API_KEY=your_key_here
```

L'évaluation peut être lancée avec :

```bash
uv run python -m app.evaluation.ragas_evaluation
```

Le script construit alors le dataset Ragas à partir du véritable système RAG puis exécute les différentes métriques.

Les appels Mistral utilisés pour l'évaluation sont donc de véritables appels externes.

---

# Tests

Le projet possède une suite de tests couvrant les principaux composants :

* API FastAPI ;
* chargement des données ;
* normalisation des événements ;
* déduplication ;
* chargement du dataset ;
* création des documents ;
* génération des embeddings ;
* construction de l'index FAISS ;
* Query Parser ;
* extraction des filtres temporels ;
* extraction des filtres géographiques ;
* absence d'inférence géographique ;
* filtrage metadata ;
* recherche vectorielle ;
* seuil de similarité ;
* `TOP_K` ;
* absence de fallback lorsque les filtres explicites ne correspondent à aucun document ;
* utilisation de la question complète pour la recherche sémantique ;
* service RAG ;
* construction du dataset Ragas ;
* orchestration de l'évaluation Ragas.

---

# Mock des appels externes

Les tests unitaires ne doivent pas dépendre d'appels réels à l'API Mistral.

Les composants externes sont donc mockés lorsque cela est nécessaire.

Par exemple, les tests du Query Parser utilisent un LLM mocké et vérifient directement la sortie structurée :

```text
Question
   ↓
Mock LLM
   ↓
EventSearchQuery
```

Cela permet :

* d'éviter les coûts API ;
* d'éviter les erreurs de rate limit ;
* d'obtenir des tests déterministes ;
* d'exécuter les tests sans connexion à Mistral.

Le retriever utilise également des index et embeddings simulés afin de tester la logique de retrieval indépendamment du service Mistral.

---

# Lancer les tests

Pour lancer toute la suite :

```bash
uv run pytest -vv
```

Pour lancer uniquement les tests du Query Parser :

```bash
uv run pytest tests/test_query_parser.py -v
```

Pour lancer uniquement les tests du retriever :

```bash
uv run pytest tests/test_retriever.py -v
```

Pour lancer uniquement les tests du service RAG :

```bash
uv run pytest tests/test_rag_service.py -v
```

---

# Résultat actuel des tests

La suite de tests actuelle contient **57 tests**, qui passent avec succès :

```text
57 passed
```

Un warning de dépréciation provenant de `Starlette` / `httpx` peut apparaître lors de l'exécution des tests, mais il ne correspond pas à une erreur du code applicatif.

---

# Couverture des tests

La couverture peut être calculée avec :

```bash
uv run pytest --cov=app --cov-report=term-missing
```

Le rapport indique :

* le pourcentage de couverture par module ;
* les lignes exécutées ;
* les lignes non couvertes.

---

# Dépendances principales

| Technologie     | Utilisation                                     |
| --------------- | ----------------------------------------------- |
| **Python 3.12** | Langage du projet                               |
| **uv**          | Gestion de l'environnement et des dépendances   |
| **Open Agenda** | Source des événements                           |
| **Mistral AI**  | Embeddings et génération de texte               |
| **FAISS**       | Recherche vectorielle                           |
| **LangChain**   | Documents, retriever et orchestration RAG       |
| **Pydantic**    | Validation et sortie structurée du Query Parser |
| **FastAPI**     | API REST                                        |
| **Uvicorn**     | Serveur ASGI                                    |
| **Ragas**       | Évaluation automatique du RAG                   |
| **pytest**      | Tests automatisés                               |

---

# Choix d'architecture

## Pourquoi utiliser un Query Parser ?

Une question utilisateur peut contenir plusieurs types d'informations :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

Certaines informations sont sémantiques :

```text
concerts
jazz
familles
```

D'autres correspondent directement à des métadonnées :

```text
ce week-end
Paris
```

Le Query Parser sépare ces deux types d'informations.

```text
Question complète
       │
       ├──────────────► Informations metadata
       │                ├── dates
       │                └── localisation
       │
       └──────────────► Informations sémantiques
                        ├── type d'événement
                        ├── thème
                        └── préférences
```

Cela permet d'utiliser les métadonnées pour un filtrage précis tout en conservant toute la richesse sémantique de la question pour la recherche vectorielle.

---

# Pourquoi filtrer les métadonnées avant FAISS ?

FAISS effectue une recherche vectorielle mais ne gère pas directement les filtres métier arbitraires tels que :

```text
location_city = Paris
date_from = 2026-09-05
date_to = 2026-09-06
```

Le système applique donc :

```text
Métadonnées
    ↓
Filtrage
    ↓
Documents candidats
    ↓
FAISS
```

Cela permet de garantir qu'un document qui ne respecte pas un filtre explicite ne sera jamais récupéré par la recherche sémantique.

---

# Pourquoi utiliser la question complète pour FAISS ?

Le Query Parser n'est pas utilisé pour reformuler la requête sémantique.

Par exemple :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

reste la requête envoyée au modèle d'embedding.

Le Query Parser extrait seulement :

```text
date_from
date_to
location_city
```

La question complète conserve :

```text
concerts
jazz
familles
```

Ces informations sont importantes pour la recherche sémantique.

Le pipeline est donc :

```text
Question complète
      │
      ├──────────────► Query Parser
      │                     │
      │                     ▼
      │                 Metadata filters
      │
      │
      └──────────────► Mistral Embedding
                            │
                            ▼
                          FAISS
```

---

# Limites du POC

Ce projet constitue un **POC (Proof of Concept)** et présente donc plusieurs limites.

## Données

Les données dépendent de l'API publique Open Agenda.

La disponibilité, la qualité et la complétude des événements dépendent donc de la source externe.

## Index vectoriel

FAISS est actuellement utilisé localement.

L'architecture ne repose pas sur une base vectorielle distribuée ou un service managé.

## Mise à jour des données

L'index doit être reconstruit pour intégrer de nouvelles données ou des modifications de la représentation des événements.

Une mise à jour incrémentale de l'index n'est pas encore implémentée.

## Recherche vectorielle

Le système utilise actuellement une recherche FAISS basée sur `IndexFlatIP`.

Il n'intègre pas encore :

* de recherche hybride BM25 + vectorielle ;
* de reranking avancé ;
* de recherche multi-vectorielle ;
* de stratégie ANN optimisée pour de très grands volumes.

## Query Parser

L'extraction des filtres temporels et géographiques dépend du LLM.

Même avec une sortie structurée Pydantic, l'interprétation des formulations complexes reste dépendante du modèle.

## Génération

La qualité des réponses dépend :

* des documents disponibles ;
* de la qualité des embeddings ;
* de la pertinence du retrieval ;
* du contexte transmis au LLM ;
* des capacités du modèle Mistral utilisé.

## Évaluation

Ragas utilise lui-même des modèles pour certaines métriques.

Les scores obtenus doivent donc être considérés comme des indicateurs de qualité et non comme une vérité absolue.

---

# Pipeline complet

Le fonctionnement complet du projet peut être résumé ainsi :

```text
                         OFFLINE / INDEXATION
                         ====================

                         Open Agenda API
                                │
                                ▼
                           Data Loader
                                │
                                ▼
                         Normalisation
                                │
                                ▼
                         Déduplication
                                │
                                ▼
                          Documents
                                │
                                ▼
                       Mistral Embeddings
                         mistral-embed
                                │
                                ▼
                             FAISS
                                │
                                ▼
                    faiss_index.bin + metadata.json


                          ONLINE / RAG
                         ============

                       Question utilisateur
                                │
                                ▼
                         Query Parser
                                │
                                ▼
                     EventSearchQuery
                                │
                                ▼
                       Metadata filters
                                │
                                ▼
                     Documents candidats
                                │
                                │
                 Question complète
                                │
                                ▼
                       Mistral Embedding
                                │
                                ▼
                              FAISS
                                │
                                ▼
                    Similarity Threshold
                                │
                                ▼
                              TOP_K
                                │
                                ▼
                      Documents pertinents
                                │
                                ▼
                       Mistral Small
                    mistral-small-latest
                                │
                                ▼
                         Réponse finale


                         ÉVALUATION
                         ==========

                     Dataset de référence
                                │
                                ▼
                       Système RAG réel
                                │
                                ├── Réponse
                                │
                                └── Contextes
                                │
                                ▼
                           Ragas
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
            Faithfulness   Answer Relevancy   Context
                                              Precision /
                                              Recall
                                │
                                ▼
                         Scores d'évaluation
```

---

# Résumé

Ce projet met en œuvre une chaîne RAG complète à partir d'événements Open Agenda :

```text
Open Agenda
     ↓
Data Loader
     ↓
Normalisation / Déduplication
     ↓
Documents LangChain
     ↓
Mistral Embeddings
     ↓
FAISS
     ↓
     ┌─────────────────────────────────┐
     │                                 │
     │        Question utilisateur     │
     │                 │               │
     │                 ▼               │
     │          Query Parser           │
     │                 │               │
     │                 ▼               │
     │        Metadata filters         │
     │                 │               │
     │                 ▼               │
     │       Documents candidats       │
     │                 │               │
     │                 ▼               │
     │    Embedding question complète  │
     │                 │               │
     │                 ▼               │
     │               FAISS             │
     │                 │               │
     │                 ▼               │
     │          Threshold + TOP_K      │
     │                 │               │
     │                 ▼               │
     │        Documents pertinents     │
     │                                 │
     └────────────────┬────────────────┘
                      │
                      ▼
                Mistral LLM
                      │
                      ▼
                Réponse finale
                      │
                      ▼
                    Ragas
                      │
                      ▼
                Évaluation RAG
```

L'architecture sépare clairement les différentes responsabilités :

```text
Data Loader
    → acquisition et préparation des données

Indexer
    → documents + embeddings + FAISS

Query Parser
    → extraction des contraintes metadata

Retriever
    → filtrage metadata + recherche vectorielle

RAG Service
    → orchestration du retrieval et de la génération

FastAPI
    → exposition du système

Ragas
    → évaluation

pytest
    → validation automatisée
```

Cette séparation permet de construire un système RAG **modulaire, testable et évolutif**, tout en conservant une architecture suffisamment simple pour analyser individuellement chaque étape du pipeline.

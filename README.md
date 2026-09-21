# POC RAG Open Agenda

Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** permettant de répondre à des questions en langage naturel à partir d'événements provenant de l'API publique **OpenAgenda**.

Le système combine :

* **OpenAgenda** pour récupérer les événements ;
* **Mistral Embeddings** (`mistral-embed`) pour transformer les événements et les questions en vecteurs ;
* **FAISS** pour effectuer la recherche vectorielle ;
* un **Query Parser** basé sur un LLM Mistral et une sortie structurée Pydantic pour extraire les contraintes temporelles et géographiques ;
* un **pré-filtrage par métadonnées** avant la recherche vectorielle ;
* un **seuil de similarité** pour éliminer les résultats trop peu pertinents ;
* **Mistral Small** (`mistral-small-latest`) pour générer la réponse finale ;
* **LangChain** pour orchestrer les documents, le retriever et la chaîne RAG ;
* **Ragas** pour évaluer automatiquement la qualité du système RAG ;
* **FastAPI** pour exposer le système sous forme d'API REST ;
* **Streamlit** pour fournir une interface utilisateur ;
* **pytest** pour tester les différents composants du projet.

L'objectif de ce POC est de mettre en œuvre une architecture RAG complète, simple, testable et compréhensible, depuis la récupération des données jusqu'à la génération et l'évaluation des réponses.

---

# Démarrage rapide

## Prérequis

* **Python 3.12**
* **uv**
* **Docker** si l'exécution conteneurisée est souhaitée
* Une clé API **Mistral AI**

## 1. Installation

Cloner le projet puis installer les dépendances :

```bash
uv sync
```

## 2. Configuration

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=your_key_here

SIMILARITY_THRESHOLD=0.45
TOP_K=3

DEFAULT_CITY=Paris
LOOKBACK_DAYS=365
```

### Variables d'environnement

| Variable               | Description                                                          | Valeur par défaut |
| ---------------------- | -------------------------------------------------------------------- | ----------------- |
| `MISTRAL_API_KEY`      | Clé API Mistral utilisée pour les embeddings, le LLM et l'évaluation | Aucune            |
| `SIMILARITY_THRESHOLD` | Seuil minimal de similarité                                          | `0.45`            |
| `TOP_K`                | Nombre maximal de documents récupérés                                | `3`               |
| `DEFAULT_CITY`         | Ville utilisée par défaut pour la récupération des événements        | `Paris`           |
| `LOOKBACK_DAYS`        | Période de récupération des événements                               | `365`             |

## 3. Lancement du projet avec Docker

Le projet est entièrement conteneurisé avec Docker. Il démarre automatiquement :

* **l'API FastAPI**, accessible sur `http://localhost:8000` ;
* **l'interface Streamlit**, accessible sur `http://localhost:8501`.

Pour construire les images et démarrer les deux services :

```bash
docker compose up --build
```

Une fois les conteneurs démarrés :

**Interface Streamlit :**

```text
http://localhost:8501
```

**API FastAPI :**

```text
http://localhost:8000
```

**Documentation Swagger :**

```text
http://localhost:8000/docs
```

### Communication entre les services

Streamlit communique avec l'API FastAPI via le réseau Docker.
La logique RAG reste entièrement dans l'API. Streamlit sert uniquement d'interface utilisateur et communique avec le backend via l'API.

Le dossier `data/` est partagé entre les deux conteneurs.
Cela permet notamment à l'API de construire ou reconstruire l'index, tandis que Streamlit peut accéder aux données nécessaires à l'affichage et à l'évaluation.

### Arrêter les conteneurs

Pour arrêter les services :

```bash
docker compose down
```

---

# Architecture générale

Le fonctionnement global du projet est le suivant :

```text
                         ┌────────────────────────┐
                         │      OpenAgenda        │
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
                         │ faiss_index.bin        │   
                         │       +                │   
                         │ metadata.json          │   
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
                    Documents candidats
                              │
                              │
                    QUESTION COMPLÈTE
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
                  Seuil de similarité + TOP_K
                              │
                              ▼
                     Documents pertinents (Scores → API)
                              │
                              ▼
                   ┌──────────────────────┐
                   │     Mistral LLM      │
                   │ mistral-small-latest │
                   └──────────┬───────────┘
                              │
                              ▼
                       Réponse finale

                       
                          ÉVALUATION
                         ============

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
            Faithfulness   Answer Relevancy   Context Precision / Recall
                                │
                                ▼
                       Scores d'évaluation
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
    → calcul du score de similarité

RAG Service

    → orchestration du retrieval et de la génération

FastAPI

    → exposition du système

Streamlit

    → interface utilisateur

Ragas

    → évaluation

pytest

    → validation automatisée
```

Cette séparation permet de construire un système RAG **modulaire, testable et évolutif**.

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
├── Dockerfile.streamlit
├── README.md
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── uv.lock
├── streamlit_app.py
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

| Module                                      | Rôle                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------- |
| `streamlit_app.py`                          | Interface utilisateur Streamlit                                      |
| `Dockerfile`                                | Construit l'image du backend FastAPI                                 |
| `Dockerfile.streamlit`                      | Construit l'image de l'interface Streamlit                           |
| `docker-compose.yml`                        | Orchestre les services API et Streamlit                              |
| `app/api.py`                                | Expose le système RAG via FastAPI                                    |
| `app/config.py`                             | Centralise la configuration et les variables d'environnement         |
| `app/core/data_loader.py`                   | Récupère, normalise et prépare les événements OpenAgenda             |
| `app/core/indexer.py`                       | Crée les documents, génère les embeddings et construit l'index FAISS |
| `app/core/query_parser.py`                  | Extrait les contraintes temporelles et géographiques de la question utilisateur                |
| `app/core/retriever.py`                     | Applique les filtres metadata puis effectue la recherche vectorielle |
| `app/core/rag_service.py`                   | Orchestre le retrieval et la génération de la réponse                |
| `app/schemas/search.py`                     | Définit les modèles Pydantic du Query Parser                         |
| `app/evaluation/dataset.py`                 | Charge les données d'évaluation                           |
| `app/evaluation/ragas_evaluation.py`        | Complète le dataset Ragas et exécute l'évaluation                   |
| `app/scripts/build_index.py`                | Script de reconstruction de l'index                                  |
| `app/scripts/resolve_reference_contexts.py` | Complète et valide les `reference_contexts` du dataset d'évaluation                          |
| `data/`                                     | Stocke l'index FAISS, les métadonnées et les données d'évaluation    |
| `tests/`                                    | Contient les tests automatisés                                       |

---

# Acquisition et préparation des données

Les événements sont récupérés depuis l'API publique OpenAgenda.

Le `Data Loader` est responsable de :

1. récupérer les événements ;
2. normaliser les différents champs ;
3. gérer les valeurs manquantes ;
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

├── page_content (contenu textuel de l'événement)
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

Le `page_content` est transformé en vecteur avec le model `mistral-embed`.

Les vecteurs sont ensuite stockés dans : `FAISS IndexFlatIP`.

`IndexFlatIP` utilise le produit scalaire (`Inner Product`) pour comparer les vecteurs.

Les embeddings utilisés dans ce projet étant normalisés, le produit scalaire permet d'obtenir une mesure correspondant à la similarité cosinus.

Les fichiers générés sont notamment :

```text
data/faiss_index.bin
data/metadata.json
```

* `faiss_index.bin` contient l'index vectoriel ;
* `metadata.json` contient les métadonnées des documents.

---

# Fonctionnement du Query Parser

Le système utilise un **Query Parser** avant la recherche vectorielle.

Son rôle est de transformer la question utilisateur en contraintes pouvant être appliquées aux métadonnées.

Le Query Parser utilise :

* `ChatPromptTemplate` de LangChain pour construire le prompt envoyé au LLM chargé d'analyser la question de l'utilisateur;
* un LLM Mistral ;
* une sortie structurée Pydantic sur le schéma `EventSearchQuery`.

Le parser extrait uniquement les contraintes temporelles et géographiques. Il **ne réécrit pas la question sémantique**.

Par exemple :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

peut produire :

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

Le Query Parser peut interpréter par exemple :

* aujourd'hui ;
* demain ;
* ce week-end ;
* la semaine dernière ;
* le mois prochain ;
* l'année prochaine ;
* une date précise ;
* une période précise ; etc..

Les expressions relatives sont converties en dates réelles à partir de la date courante.

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

## Pourquoi utiliser un Query Parser pour filtrer avant FAISS ?

FAISS réalise une recherche sémantique, mais ne prend pas en charge directement les filtres sur les métadonnées.

Les documents sont donc d'abord filtrés selon les critères explicites de la question, puis la recherche vectorielle est effectuée uniquement parmi les documents correspondants.

Cela garantit que la demande utilisateur soit toujours respectés.

---

# Fonctionnement du Retrieval

Le retrieval utilise une architecture en deux étapes :

```text
Question utilisateur
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
  Query Parser                 Question originale
        │                             │
        ▼                             │
 Filtres metadata                     │
        │                             │
        ▼                             │
Documents candidats                   │
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
        Embedding de la question complète
                       │
                       ▼
                Recherche FAISS
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

## 2. Pré-filtrage des Documents avec les metadata

Les métadonnées sont ensuite utilisées pour sélectionner les documents candidats.

Par exemple :

```text
Tous les événements
       ↓
Ville = Paris
       ↓
Date compatible avec la période demandée
       ↓
Documents candidats
```

Remarque : Un événement est conservé lorsqu'il **chevauche la période recherchée**. Cela permet notamment de conserver un événement qui commence avant la période demandée mais se poursuit pendant celle-ci.


Un événement doit donc respecter l'ensemble des contraintes explicites (filtres combinés avec une logique `AND`).


## 3. Embedding de la question utilsiateur

Une fois les documents candidats déterminés, la **question complète de l'utilisateur** est transformée en embedding avec `mistral-embed`.

La question contient les informations sémantiques nécessaires à la recherche.

Par exemple :

```text
Quels concerts de jazz sont prévus ce week-end à Paris pour les familles ?
```

est la requête envoyée au modèle d'embedding.

Le Query Parser extrait les contraintes, mais ne remplace pas la question originale.

## 4. Recherche FAISS

FAISS compare l'embedding de la question avec les embeddings des documents candidats.

Deux cas sont possibles :
1) Aucun filtre metadata. La recherche est effectuée directement dans l'index FAISS global.
2) Présence de filtres metadata. Seuls les documents correspondant aux filtres sont utilisés comme candidats. Le retriever construit alors une recherche FAISS temporaire à partir des embeddings des documents candidats. Cela garantit qu'un document qui ne respecte pas une contrainte explicite n'est pas récupéré.

## 5. Seuil de similarité

Les résultats dont le score de similarité est inférieur à :

```env
SIMILARITY_THRESHOLD=0.45
```

sont supprimés.

Un résultat dont le score est exactement égal au seuil est conservé.

## 6. TOP_K

Le paramètre :

```env
TOP_K=3
```

limite le nombre maximal de documents récupérés.

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

Le nombre réel peut être inférieur à `TOP_K` si :

* moins de documents candidats existent ;
* certains documents sont sous le seuil de similarité.

## Aucun fallback sur les filtres explicites

Lorsqu'un utilisateur fournit un filtre explicite, le système **ne revient pas à une recherche globale** si aucun document ne correspond.

Par exemple :

```text
Quels événements sont prévus à Bordeaux ?
```

Si aucun événement correspond à :

```text
location_city = Bordeaux
```

le retriever retourne zéro document.

Il ne recherche pas automatiquement dans tous les événements.

Cela évite de retourner des événements qui ne respectent pas la contrainte demandée.

## Gestion des scores de retrieval

Le retriever calcule un score de similarité pour chaque document récupéré et le stocke séparément dans le résultat de recherche, sans l'ajouter aux Document.metadata.

Le score est une information technique utile pour :

* l'API ;
* l'interface utilisateur ;
* l'analyse du retrieval ;
* les tests ;
* l'évaluation.

Il n'est cependant pas nécessaire au LLM pour générer la réponse.

---

# Génération de la réponse

Après le retrieval, les documents pertinents sont transmis au LLM `mistral-small-latest`.

Le LLM reçoit :

* la question utilisateur;
* les documents récupérés comme contexte.

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

L'objectif est de faire générer la réponse à partir des informations récupérées par le système RAG.

Si aucun document suffisamment pertinent n'est trouvé, aucun contexte documentaire pertinent n'est transmis au LLM.

---

# Architecture LangChain

**LangChain** est un framework Python qui permet de construire et d'organiser des applications utilisant des LLM. Il fournit notamment des interfaces standardisées pour les **retrievers**, les **documents**, les **prompts** et les **chaînes de traitement**.

Dans notre projet, LangChain est principalement utilisé pour **structurer le pipeline RAG** et faire communiquer ses différentes étapes de manière standardisée. Cela facilite notamment l'intégration avec les composants de recherche et de génération, ainsi que les tests et l'évolution de l'architecture.

Notre `EventRetriever` implémente l'interface `BaseRetriever` de LangChain. Il réalise la recherche des documents en utilisant les différents composants du pipeline RAG :

```text
EventRAGService
       │
       ▼
EventRetriever (BaseRetriever)
       │
       ├── Query Parser
       │
       ├── Filtrage des metadata
       │
       ├── Embedding Mistral
       │
       ├── Recherche FAISS
       │
       ├── Seuil de similarité
       │
       └── TOP_K
       │
       ▼
Liste de Documents LangChain
       │
       ▼
Document chain
       │
       ▼
LLM Mistral
       │
       ▼
Réponse
```

Dans notre cas, LangChain nous permet donc notamment de :

* représenter les événements récupérés sous forme de `Document` ;
* intégrer notre `EventRetriever` dans le pipeline RAG grâce à `BaseRetriever` ;
* construire le prompt envoyé au LLM avec `ChatPromptTemplate` ;
* transmettre les documents pertinents au LLM via la `Document chain`.

La séparation des responsabilités permet de tester et de faire évoluer indépendamment :

* le parsing de la question ;
* le filtrage des metadata ;
* la recherche vectorielle ;
* le retriever ;
* le service RAG ;
* la génération de la réponse ;
* l'évaluation.

---

# API FastAPI

L'application est exposée via FastAPI.

# Endpoints principaux

## `GET /`

Endpoint de vérification de disponibilité de l'API.

```bash
curl http://127.0.0.1:8000/
```

Réponse :

```json
{
  "status": "ok"
}
```

## `POST /ask`

Cet endpoint permet de poser une question au système RAG et de retourner la réponse générée ainsi que les documents récupérés avec leurs scores de similarité.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels concerts de jazz sont prévus ce week-end à Paris ?"}'
```

La réponse de l'API contient :

```json
{
  "answer": "...",
  "used_mistral": true,
  "context": [
    {
      "score": 0.91,
      "document": "..."
    }
  ]
}
```

Les scores sont donc disponibles au niveau de l'API, mais ils ne sont pas intégrés aux métadonnées des documents et ne sont pas transmis au LLM.


## `POST /rebuild`

Cet endpoint permet de reconstruire l'index du système RAG à partir des données OpenAgenda.

La reconstruction récupère les événements, prépare les données, génère les embeddings avec `mistral-embed`, reconstruit et sauvegarde l'index FAISS et les métadonnées.


Exemple :

```bash
curl -X POST http://127.0.0.1:8000/rebuild
```

Il est nécessaire de reconstruire l'index lorsque les données ou leur représentation changent.

Par exemple :

* nouveaux événements OpenAgenda ;
* modification de la préparation du `page_content` ;
* modification des métadonnées indexées ;
* changement du modèle d'embedding ;
* modification de la génération des embeddings.

## `POST /evaluate`

Endpoint qui lance l'évaluation du système RAG avec **Ragas**.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/evaluate
```

L'évaluation exécute le système RAG sur le dataset de référence, récupère les réponses et les contextes, puis calcule les métriques :

* **Faithfulness**
* **Answer Relevancy**
* **Context Precision**
* **Context Recall**

La réponse de l'endpoint contient notamment les métriques moyennes, les scores obtenus pour chaque question et les résultats détaillés de l'évaluation.

L'évaluation peut prendre plusieurs minutes.

---

# Interface Streamlit

L'interface contient notamment :
1) Une page Chat de conversation avec le système RAG qui permet de :
    * poser une question ;
    * lancer `/ask` ;
    * afficher la réponse généréé;
    * afficher l'historique de conversation ;

2) Une page d'évaluation qui permet de :
    * lancer `/evaluate` ;
    * afficher les métriques Ragas ;
    * visualiser les résultats ;
    * consulter les détails par question.

---

# Pipeline d'évaluation

Le projet contient une évaluation automatique du système RAG avec **Ragas**.

Le dataset de référence se trouve dans `data/evaluation/rag_evaluation.json` et contient :

```json
{
  "user_input": "Quels événements sont prévus prochainement à Paris ?",
  "reference": "Voici 3 événements prévus prochainement à Paris : une exposition, ...",
  "reference_doc_ids": [
    "49229737"
  ],
  "reference_contexts": [
    "Titre: De la sauvegarde à la conservation-restauration du patrimoine cinématographique\nDate: 2026-09-19\nLieu: Cinémathèque française\nVille: Paris"
  ],
  "response": "Voici les événements prévus prochainement à Paris : ...",
  "retrieved_contexts": [
    "..."
  ]
}
```

Les principaux champs sont :

* `user_input` : **la question posée par l'utilisateur** au système RAG.
* `reference` : **la réponse de référence attendue**, utilisée pour évaluer la réponse générée.
* `reference_doc_ids` : **les identifiants des documents considérés comme pertinents pour répondre à la question**.
* `reference_contexts` : **les contenus de référence associés aux documents pertinents**. Ils servent notamment à évaluer si le système a retrouvé les bons contextes.
* `response` : **la réponse réellement générée par le système RAG**.
* `retrieved_contexts` : **les documents réellement récupérés par le retriever** pour répondre à la question.

Cela permet à Ragas de comparer **la question, la réponse générée et les contextes récupérés** avec les références attendues afin de calculer les différentes métriques d'évaluation.


Le processus d'évaluation est le suivant :

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
Dataset Ragas
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

Mesure dans quelle mesure la réponse générée est supportée par les contextes récupérés.

### Answer Relevancy

Mesure la pertinence de la réponse par rapport à la question utilisateur.

### Context Precision

Évalue la pertinence des documents récupérés par rapport à la question.

### Context Recall

Évalue si les informations nécessaires à la réponse sont présentes dans les contextes récupérés.

---

# Tests

Le projet possède une suite de tests couvrant notamment :

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
* séparation des scores de retrieval et des `Document.metadata` ;
* transmission des scores à l'API uniquement ;
* service RAG ;
* construction du dataset Ragas ;
* orchestration de l'évaluation Ragas.

## Mock des appels externes

Les tests unitaires ne doivent pas dépendre d'appels réels à l'API Mistral.

Les composants externes sont donc mockés lorsque cela est nécessaire.

Par exemple, les tests du Query Parser utilisent un LLM mocké :

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

Le retriever utilise également des index et embeddings simulés afin de tester la logique de retrieval indépendamment de Mistral.


## Lancer les tests

Lancer toute la suite :

```bash
uv run pytest -vv
```

Lancer uniquement les tests du Query Parser :

```bash
uv run pytest tests/test_query_parser.py -v
```

Lancer uniquement les tests du retriever :

```bash
uv run pytest tests/test_retriever.py -v
```

Lancer uniquement les tests du service RAG :

```bash
uv run pytest tests/test_rag_service.py -v
```

## Couverture des tests

La couverture peut être calculée avec :

```bash
uv run pytest --cov=app --cov-report=term-missing
```

Le rapport indique :

* le pourcentage de couverture par module ;
* les lignes exécutées ;
* les lignes non couvertes.

---

# Limites du POC

Ce projet constitue un **POC (Proof of Concept)** et présente donc plusieurs limites.

## Données

Les données dépendent de l'API publique OpenAgenda.

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

La qualité des réponses dépend notamment :

* des documents disponibles ;
* de la qualité des embeddings ;
* de la pertinence du retrieval ;
* du contexte transmis au LLM ;
* des capacités du modèle Mistral utilisé.

## Évaluation

Ragas utilise lui-même des modèles pour certaines métriques.

Les scores obtenus doivent donc être considérés comme des indicateurs de qualité et non comme une vérité absolue.


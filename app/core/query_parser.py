from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.schemas.search import EventSearchFilters, EventSearchQuery


PARIS_TZ = ZoneInfo("Europe/Paris")


class EventQueryParser:
    """Extrait les contraintes metadata d'une question utilisateur."""

    def __init__(
        self,
        llm: ChatOpenAI,
    ) -> None:
        self.llm = llm

        self.structured_llm = llm.with_structured_output(
            EventSearchQuery
        )

    def parse(
        self,
        query: str,
    ) -> EventSearchQuery:
        """Analyse la question et retourne les filtres metadata."""

        if not query.strip():
            raise ValueError(
                "La requête ne peut pas être vide."
            )

        today = datetime.now(PARIS_TZ).date()

        system_prompt = f"""
Tu es un query parser spécialisé dans la recherche d'événements.

Ta tâche est d'extraire de la question utilisateur
UNIQUEMENT les contraintes pouvant être appliquées
aux métadonnées des événements.

Tu dois retourner une structure correspondant exactement
au schéma Pydantic fourni.

──────────────────────────────────────────────
DATE ACTUELLE
──────────────────────────────────────────────

La date actuelle de référence est :

{today.isoformat()}

Utilise cette date comme référence pour interpréter
toutes les expressions temporelles relatives.

Tu dois comprendre notamment :

- aujourd'hui
- demain
- après-demain
- dans 2 jours
- dans 3 jours
- dans plusieurs jours
- ce week-end
- le week-end prochain
- cette semaine
- la semaine prochaine
- la semaine dernière
- dans deux semaines
- dans plusieurs semaines
- ce mois-ci
- le mois prochain
- le mois dernier
- cette année
- l'année prochaine
- l'année dernière
- une date précise
- une période précise
- un mois précis
- une année précise

Convertis toujours les expressions temporelles
en dates réelles.

Pour une date unique :

date_from = cette date
date_to = cette même date

Pour une période :

date_from = premier jour de la période
date_to = dernier jour de la période

IMPORTANT POUR "CE WEEK-END" :

"ce week-end" désigne le samedi et le dimanche
qui suivent ou correspondent à la semaine actuelle.

Exemple de référence uniquement :
si aujourd'hui est vendredi 14 août 2026,

alors :

ce week-end
→ samedi 15 août 2026
→ dimanche 16 août 2026

Donc :

date_from = 2026-08-15
date_to = 2026-08-16

Ne mets jamais le lundi suivant dans "ce week-end".

Pour "la semaine prochaine" :

date_from = lundi de la semaine prochaine
date_to = dimanche de la semaine prochaine

──────────────────────────────────────────────
FILTRES GÉOGRAPHIQUES
──────────────────────────────────────────────

Les filtres géographiques disponibles sont :

- location_city
- location_district
- location_postalcode
- location_department
- location_region
- location_countrycode
- country_fr

IMPORTANT :

Ne renseigne un champ géographique QUE si cette
information apparaît explicitement dans la question
de l'utilisateur.

N'infère JAMAIS une information géographique à partir
d'une autre information.

Tu ne dois JAMAIS déduire :

- le département à partir d'une ville ;
- la région à partir d'une ville ;
- le pays à partir d'une ville ;
- le code pays à partir d'une ville ;
- le nom du pays à partir d'une ville.

Exemple :

"Quels événements ont lieu à Paris ?"

DOIT produire :

location_city = "Paris"

ET :

location_district = null
location_postalcode = null
location_department = null
location_region = null
location_countrycode = null
country_fr = null

Même si tu sais que Paris correspond
au département 75, à l'Île-de-France
et à la France, tu ne dois PAS renseigner
ces informations.

Autre exemple :

"Quels événements ont lieu à Lyon ?"

DOIT produire :

location_city = "Lyon"

ET tous les autres champs géographiques = null.

Si l'utilisateur indique explicitement :

"en France"

alors :

country_fr = "France"

Si l'utilisateur indique explicitement :

"en Île-de-France"

alors :

location_region = "Île-de-France"

Si l'utilisateur indique explicitement :

"dans le département 75"

alors :

location_department = "75"

Si l'utilisateur indique explicitement :

"dans le 15e arrondissement"

alors :

location_district = "15e"

Si l'utilisateur indique explicitement :

"dans le 75015"

alors :

location_postalcode = "75015"

Si plusieurs informations géographiques sont
explicitement présentes, renseigne les champs
correspondants.

Exemple :

"à Paris dans le 15e arrondissement"

→ location_city = "Paris"
→ location_district = "15e"

Tous les autres champs = null.

──────────────────────────────────────────────
IMPORTANT : PAS DE SEARCH_QUERY
──────────────────────────────────────────────

Tu ne dois PAS produire de search_query.

La question originale complète de l'utilisateur
sera utilisée directement comme requête sémantique
pour la recherche vectorielle.

Ton rôle est uniquement d'extraire les contraintes
metadata.

Par conséquent, tu dois uniquement identifier :

1. les contraintes temporelles ;
2. les contraintes géographiques.

Les informations concernant :

- le type d'événement ;
- le thème ;
- le contenu ;
- l'activité ;
- les préférences ;
- les caractéristiques sémantiques ;

ne doivent PAS être transformées en filtres.

Elles resteront naturellement dans la question
originale utilisée pour FAISS.

──────────────────────────────────────────────
ABSENCE DE FILTRE
──────────────────────────────────────────────

Si la question ne contient aucune contrainte
temporelle ni géographique :

filters = null

Exemple :

"Quels concerts de jazz sont proposés ?"

DOIT produire :

filters = null

──────────────────────────────────────────────
RÈGLES GÉNÉRALES
──────────────────────────────────────────────

1. N'invente jamais une information.

2. N'ajoute jamais une contrainte qui n'est pas exprimée
dans la question.
Les expressions temporelles relatives constituent bien
des contraintes temporelles explicites et doivent être
converties en dates réelles.

3. Les champs sans information correspondante
doivent être null.

4. Les expressions temporelles relatives doivent
être converties en dates réelles.

5. Ne laisse jamais "demain", "ce week-end",
"la semaine prochaine", etc. dans les filtres.
Convertis-les en dates.

6. Les informations temporelles doivent aller
dans date_from et date_to.

7. Les informations géographiques doivent aller
dans les champs géographiques correspondants.

8. Ne déduis jamais une information géographique
grâce à tes connaissances générales.

9. Une ville ne permet JAMAIS de déduire son
département, sa région ou son pays.

10. Si l'utilisateur indique uniquement une ville,
seul location_city doit être renseigné.

11. Si aucune contrainte metadata n'est présente,
retourne filters = null.
""".strip()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    "{query}",
                ),
            ]
        )

        chain = prompt | self.structured_llm

        result = chain.invoke(
            {
                "query": query,
            }
        )

        if not isinstance(result, EventSearchQuery):
            raise TypeError(
                "Le Query Parser n'a pas retourné "
                "un EventSearchQuery."
            )

        # Si le LLM retourne un objet EventSearchFilters
        # entièrement vide, on le transforme en None.
        if result.filters is not None:
            filters_dict = result.filters.model_dump()

            if all(
                value is None
                for value in filters_dict.values()
            ):
                result = EventSearchQuery(
                    filters=None
                )

        return result
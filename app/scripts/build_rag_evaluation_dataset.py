"""Construit un dataset d'évaluation RAG à partir des données OpenAgenda."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import DEFAULT_CITY
from app.core.data_loader import fetch_events_from_openagenda


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("data/evaluation/rag_evaluation.json")
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET_SIZE = 25


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def clean(value: Any) -> str:
    """Convertit une valeur en texte propre."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "nat"}:
        return ""

    return text


def event_title(event: dict[str, Any]) -> str:
    """Retourne le meilleur titre disponible."""
    title = clean(event.get("title_fr"))

    if title:
        return title

    slug = clean(event.get("slug"))

    if slug:
        return slug

    return "Événement sans titre"


def event_date(event: dict[str, Any]) -> str:
    """Retourne la date de début sous une forme lisible."""
    raw = clean(event.get("firstdate_begin"))

    if not raw:
        return "date inconnue"

    try:
        parsed = datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        )

        return parsed.strftime("%d/%m/%Y")

    except ValueError:
        return raw[:10]


def event_date_obj(event: dict[str, Any]) -> date | None:
    """Retourne la date de début sous forme de date."""
    raw = clean(event.get("firstdate_begin"))

    if not raw:
        return None

    try:
        return datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        ).date()

    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def event_location(event: dict[str, Any]) -> str:
    """Retourne le lieu de l'événement."""
    location_name = clean(event.get("location_name"))
    address = clean(event.get("location_address"))

    if location_name and address:
        return f"{location_name}, {address}"

    if location_name:
        return location_name

    if address:
        return address

    return "lieu non renseigné"


def event_summary(event: dict[str, Any]) -> str:
    """Construit une référence factuelle courte."""
    return (
        f"{event_title(event)} "
        f"({event_date(event)}, "
        f"{event_location(event)})"
    )


def contains_keywords(
    event: dict[str, Any],
    keywords: list[str],
) -> bool:
    """Cherche des mots-clés dans les champs textuels de l'événement."""

    fields = [
        event.get("title_fr"),
        event.get("description_fr"),
        event.get("longdescription_fr"),
        event.get("keywords_fr"),
        event.get("conditions_fr"),
    ]

    text = " ".join(
        clean(value).lower()
        for value in fields
    )

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def first_matching_event(
    events: list[dict[str, Any]],
    keywords: list[str],
    used_uids: set[str],
) -> dict[str, Any] | None:
    """Retourne le premier événement correspondant aux mots-clés."""

    for event in events:
        uid = clean(event.get("uid"))

        if uid in used_uids:
            continue

        if contains_keywords(event, keywords):
            return event

    return None


def events_matching(
    events: list[dict[str, Any]],
    keywords: list[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retourne plusieurs événements correspondant aux mots-clés."""

    matches = []

    for event in events:
        if contains_keywords(event, keywords):
            matches.append(event)

        if len(matches) >= limit:
            break

    return matches


def reference_from_events(
    events: list[dict[str, Any]],
) -> str:
    """Construit une référence à partir d'événements réels."""

    if not events:
        return ""

    return " ; ".join(
        event_summary(event)
        for event in events
    )


def next_month_name(month: int) -> str:
    """Retourne le nom français du mois."""
    months = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]

    return months[month - 1]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """Retourne le premier et le dernier jour d'un mois."""

    start = date(year, month, 1)

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    end = next_month - timedelta(days=1)

    return start, end


def events_in_period(
    events: list[dict[str, Any]],
    start: date,
    end: date,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retourne les événements dont la date de début est dans la période."""

    matches = []

    for event in events:
        event_date_value = event_date_obj(event)

        if event_date_value is None:
            continue

        if start <= event_date_value <= end:
            matches.append(event)

        if len(matches) >= limit:
            break

    return matches


# ---------------------------------------------------------------------------
# Construction du dataset
# ---------------------------------------------------------------------------

def build_dataset(
    events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Construit un dataset d'évaluation diversifié."""

    if not events:
        raise RuntimeError(
            "Aucun événement disponible pour construire "
            "le dataset d'évaluation."
        )

    city = DEFAULT_CITY
    today = date.today()

    dataset: list[dict[str, str]] = []

    used_uids: set[str] = set()

    def add_question(
        question: str,
        matching_events: list[dict[str, Any]],
    ) -> None:
        """Ajoute une question si elle possède une référence réelle."""

        if not matching_events:
            return

        dataset.append(
            {
                "user_input": question,
                "reference": reference_from_events(
                    matching_events
                ),
            }
        )

        for event in matching_events:
            uid = clean(event.get("uid"))

            if uid:
                used_uids.add(uid)

    # ------------------------------------------------------------------
    # 1. Recherche générale par ville
    # ------------------------------------------------------------------

    add_question(
        f"Quels événements sont organisés à {city} ?",
        events[:3],
    )

    # ------------------------------------------------------------------
    # 2. Événements à venir
    # ------------------------------------------------------------------

    upcoming = [
        event
        for event in events
        if (
            event_date_obj(event) is not None
            and event_date_obj(event) >= today
        )
    ]

    add_question(
        f"Quels événements sont prévus prochainement à {city} ?",
        upcoming[:3],
    )

    # ------------------------------------------------------------------
    # 3. Demain
    # ------------------------------------------------------------------

    tomorrow = events_in_period(
        events,
        today + timedelta(days=1),
        today + timedelta(days=1),
    )

    add_question(
        f"Quels événements sont prévus demain à {city} ?",
        tomorrow,
    )

    # ------------------------------------------------------------------
    # 4. Dans deux jours
    # ------------------------------------------------------------------

    in_two_days = events_in_period(
        events,
        today + timedelta(days=2),
        today + timedelta(days=2),
    )

    add_question(
        f"Quels événements auront lieu dans 2 jours à {city} ?",
        in_two_days,
    )

    # ------------------------------------------------------------------
    # 5. Cette semaine
    # ------------------------------------------------------------------

    week_start = today - timedelta(
        days=today.weekday()
    )

    week_end = week_start + timedelta(days=6)

    this_week = events_in_period(
        events,
        week_start,
        week_end,
    )

    add_question(
        f"Quels événements ont lieu cette semaine à {city} ?",
        this_week,
    )

    # ------------------------------------------------------------------
    # 6. Semaine prochaine
    # ------------------------------------------------------------------

    next_week_start = week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)

    next_week = events_in_period(
        events,
        next_week_start,
        next_week_end,
    )

    add_question(
        f"Quels événements sont prévus la semaine prochaine à {city} ?",
        next_week,
    )

    # ------------------------------------------------------------------
    # 7. Ce week-end
    # ------------------------------------------------------------------

    days_until_saturday = 5 - today.weekday()

    if days_until_saturday < 0:
        days_until_saturday += 7

    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)

    weekend = events_in_period(
        events,
        saturday,
        sunday,
    )

    add_question(
        f"Quels événements ont lieu ce week-end à {city} ?",
        weekend,
    )

    # ------------------------------------------------------------------
    # 8. Mois courant
    # ------------------------------------------------------------------

    current_month_start, current_month_end = month_bounds(
        today.year,
        today.month,
    )

    current_month_events = events_in_period(
        events,
        current_month_start,
        current_month_end,
    )

    add_question(
        f"Quels événements ont lieu à {city} en "
        f"{next_month_name(today.month)} ?",
        current_month_events,
    )

    # ------------------------------------------------------------------
    # 9. Concerts
    # ------------------------------------------------------------------

    concerts = events_matching(
        events,
        [
            "concert",
            "musique",
            "musical",
        ],
    )

    add_question(
        f"Y a-t-il des concerts à {city} ?",
        concerts,
    )

    # ------------------------------------------------------------------
    # 10. Spectacles
    # ------------------------------------------------------------------

    shows = events_matching(
        events,
        [
            "spectacle",
            "théâtre",
            "cirque",
        ],
    )

    add_question(
        f"Quels spectacles sont proposés à {city} ?",
        shows,
    )

    # ------------------------------------------------------------------
    # 11. Culture
    # ------------------------------------------------------------------

    cultural = events_matching(
        events,
        [
            "culture",
            "exposition",
            "musée",
            "cinéma",
            "art",
            "patrimoine",
        ],
    )

    add_question(
        f"Quels événements culturels sont proposés à {city} ?",
        cultural,
    )

    # ------------------------------------------------------------------
    # 12. Sport
    # ------------------------------------------------------------------

    sport = events_matching(
        events,
        [
            "sport",
            "football",
            "basket",
            "tennis",
            "course",
            "randonnée",
        ],
    )

    add_question(
        f"Quels événements sportifs sont proposés à {city} ?",
        sport,
    )

    # ------------------------------------------------------------------
    # 13. Famille / enfants
    # ------------------------------------------------------------------

    family = events_matching(
        events,
        [
            "famille",
            "familial",
            "enfant",
            "enfants",
            "jeunesse",
            "jeune public",
        ],
    )

    add_question(
        f"Quels événements sont adaptés aux enfants à {city} ?",
        family,
    )

    # ------------------------------------------------------------------
    # 14. Gratuit
    # ------------------------------------------------------------------

    free = events_matching(
        events,
        [
            "gratuit",
            "gratuite",
            "gratuitement",
            "entrée libre",
        ],
    )

    add_question(
        f"Quels événements gratuits sont proposés à {city} ?",
        free,
    )

    # ------------------------------------------------------------------
    # 15. Premier événement spécifique
    # ------------------------------------------------------------------

    first_event = events[0]

    add_question(
        f"Que peux-tu me dire sur l'événement "
        f"« {event_title(first_event)} » ?",
        [first_event],
    )

    # ------------------------------------------------------------------
    # 16. Deuxième événement spécifique
    # ------------------------------------------------------------------

    if len(events) > 1:
        second_event = events[1]

        add_question(
            f"À quelle date a lieu "
            f"« {event_title(second_event)} » ?",
            [second_event],
        )

    # ------------------------------------------------------------------
    # 17. Lieu d'un événement
    # ------------------------------------------------------------------

    if len(events) > 2:
        third_event = events[2]

        add_question(
            f"Où a lieu "
            f"« {event_title(third_event)} » ?",
            [third_event],
        )

    # ------------------------------------------------------------------
    # 18. Quartier
    # ------------------------------------------------------------------

    district_events = [
        event
        for event in events
        if clean(event.get("location_district"))
    ]

    if district_events:
        district = clean(
            district_events[0].get("location_district")
        )

        district_matches = [
            event
            for event in events
            if clean(event.get("location_district")) == district
        ][:3]

        add_question(
            f"Quels événements ont lieu dans le quartier "
            f"{district} à {city} ?",
            district_matches,
        )

    # ------------------------------------------------------------------
    # 19. Code postal
    # ------------------------------------------------------------------

    postal_events = [
        event
        for event in events
        if clean(event.get("location_postalcode"))
    ]

    if postal_events:
        postalcode = clean(
            postal_events[0].get("location_postalcode")
        )

        postal_matches = [
            event
            for event in events
            if clean(event.get("location_postalcode"))
            == postalcode
        ][:3]

        add_question(
            f"Quels événements sont organisés dans le "
            f"{postalcode} ?",
            postal_matches,
        )

    # ------------------------------------------------------------------
    # 20. Région
    # ------------------------------------------------------------------

    region_events = [
        event
        for event in events
        if clean(event.get("location_region"))
    ]

    if region_events:
        region = clean(
            region_events[0].get("location_region")
        )

        region_matches = [
            event
            for event in events
            if clean(event.get("location_region"))
            == region
        ][:3]

        add_question(
            f"Quels événements sont proposés en "
            f"{region} ?",
            region_matches,
        )

    # ------------------------------------------------------------------
    # 21. Pays
    # ------------------------------------------------------------------

    country_events = [
        event
        for event in events
        if clean(event.get("country_fr"))
    ]

    if country_events:
        country = clean(
            country_events[0].get("country_fr")
        )

        country_matches = [
            event
            for event in events
            if clean(event.get("country_fr"))
            == country
        ][:3]

        add_question(
            f"Quels événements sont organisés en "
            f"{country} ?",
            country_matches,
        )

    # ------------------------------------------------------------------
    # 22. Adresse
    # ------------------------------------------------------------------

    address_events = [
        event
        for event in events
        if clean(event.get("location_address"))
    ]

    if address_events:
        address_event = address_events[0]

        add_question(
            f"À quelle adresse a lieu "
            f"« {event_title(address_event)} » ?",
            [address_event],
        )

    # ------------------------------------------------------------------
    # 23. Mots-clés d'un événement réel
    # ------------------------------------------------------------------

    keyword_event = next(
        (
            event
            for event in events
            if clean(event.get("keywords_fr"))
        ),
        None,
    )

    if keyword_event:
        keywords = clean(
            keyword_event.get("keywords_fr")
        )

        first_keyword = re.split(
            r"[,;|]",
            keywords,
        )[0].strip()

        if first_keyword:
            keyword_matches = events_matching(
                events,
                [first_keyword],
            )

            add_question(
                f"Quels événements liés à "
                f"{first_keyword} sont proposés à {city} ?",
                keyword_matches,
            )

    # ------------------------------------------------------------------
    # 24. Description / activité
    # ------------------------------------------------------------------

    description_event = next(
        (
            event
            for event in events
            if clean(event.get("description_fr"))
        ),
        None,
    )

    if description_event:
        add_question(
            f"Peux-tu me recommander un événement intéressant "
            f"à {city} ?",
            [description_event],
        )

    # ------------------------------------------------------------------
    # 25. Question sémantique générale
    # ------------------------------------------------------------------

    semantic_event = next(
        (
            event
            for event in events
            if clean(event.get("title_fr"))
            and clean(event.get("description_fr"))
        ),
        None,
    )

    if semantic_event:
        add_question(
            f"Que peut-on faire prochainement à {city} ?",
            [semantic_event],
        )

    # ------------------------------------------------------------------
    # Nettoyage final
    # ------------------------------------------------------------------

    # On supprime les doublons de questions.
    unique_dataset: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for item in dataset:
        question = item["user_input"]

        if question in seen_questions:
            continue

        seen_questions.add(question)
        unique_dataset.append(item)

    # Si certaines catégories n'existent pas dans les données,
    # on complète avec des questions portant sur des événements
    # réels afin de garantir TARGET_SIZE.
    fallback_index = 0

    while (
        len(unique_dataset) < TARGET_SIZE
        and fallback_index < len(events)
    ):
        event = events[fallback_index]

        question = (
            f"Peux-tu me donner les informations disponibles "
            f"sur l'événement « {event_title(event)} » ?"
        )

        if question not in seen_questions:
            unique_dataset.append(
                {
                    "user_input": question,
                    "reference": event_summary(event),
                }
            )

            seen_questions.add(question)

        fallback_index += 1

    return unique_dataset[:TARGET_SIZE]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Récupère OpenAgenda et construit le dataset."""

    print("\n========================================")
    print("BUILD RAG EVALUATION DATASET")
    print("========================================\n")

    print(
        f"[INFO] Récupération des événements "
        f"pour {DEFAULT_CITY}..."
    )

    events = fetch_events_from_openagenda()

    print(
        f"[INFO] Événements récupérés : {len(events)}"
    )

    if not events:
        raise RuntimeError(
            "Aucun événement récupéré depuis OpenAgenda."
        )

    dataset = build_dataset(events)

    print(
        f"[INFO] Questions générées : {len(dataset)}"
    )

    if len(dataset) < TARGET_SIZE:
        print(
            "[WARNING] Impossible de générer exactement "
            f"{TARGET_SIZE} questions avec les données actuelles."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[INFO] Dataset sauvegardé dans : "
        f"{OUTPUT_PATH}"
    )

    print("\nQuestions générées :\n")

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"{index:02d}. {item['user_input']}"
        )

    print("\n========================================")
    print("DONE")
    print("========================================\n")


if __name__ == "__main__":
    main()
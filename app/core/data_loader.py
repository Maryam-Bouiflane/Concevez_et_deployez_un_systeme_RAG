"""Fonctions pour charger des événements depuis l'API Open Agenda."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from app.config import DEFAULT_CITY, LOOKBACK_DAYS
from datetime import date, timedelta

def add_label(series: pd.Series, label: str) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.where(values.eq(""), label + values)

def fetch_events_from_openagenda(city: str = DEFAULT_CITY, lookback_days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Récupère des événements récents depuis l'API Open Agenda."""
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records" 
    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    all_records = []
    limit = 100
    offset = 0
    max_pages = 100

    try:
        for _ in range(max_pages):
            params = {
                "where": (
                    f"location_city = '{city}' "
                    f"AND lastdate_end >= '{start_date}'"
                ),
                "limit": limit,
                "offset": offset,
                "timezone": "Europe/Paris",
            }

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()

            batch = response.json().get("results", [])

            if not batch:
                break

            all_records.extend(batch)

            if len(batch) < limit:
                break

            offset += limit

        if not all_records:
            return []

        df = pd.json_normalize(all_records)

        # Titre avec fallback sur le slug
        title = (
            df["title_fr"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        slug = (
            df["slug"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        title = title.where(title.ne(""), slug)

        # Dates
        firstdate = (
            df["firstdate_begin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        lastdate = (
            df["lastdate_end"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        date_text = firstdate.where(
            lastdate.eq(""),
            firstdate + " - " + lastdate,
        ).where(
            firstdate.ne(""),
            lastdate,
        )

        # Texte destiné à la vectorisation
        text_fields = [
            add_label(title, "Titre: "),
            add_label(df["description_fr"], "Description: "),
            add_label(df["longdescription_fr"], "Détails: "),
            add_label(df["keywords_fr"], "Mots-clés: "),
            # Dates
            add_label(date_text, "Date: "),
            # Informations sur le lieu
            add_label(df["location_name"], "Lieu: "),
            add_label(df["location_address"], "Adresse: "),
            add_label(df["location_city"], "Ville: "),
            add_label(df["location_district"], "Quartier: "),
            add_label(df["location_postalcode"], "Code postal: "),
            add_label(df["location_department"], "Département: "),
            add_label(df["location_region"], "Région: "),
            add_label(df["country_fr"], "Pays: "),
        ]

        df["text"] = (
            pd.concat(text_fields, axis=1)
            .apply(
                lambda row: "\n".join(value for value in row if value),
                axis=1,
            )
        )
        # Laisser text à None si aucun champ n'est présent
        df.loc[df["text"].str.strip().eq(""), "text"] = None

        relevant_columns = [
            "uid",
            "canonicalurl",
            "slug",
            "text",
            "title_fr",
            "description_fr",
            "longdescription_fr",
            "conditions_fr",
            "keywords_fr",
            # "daterange_fr",
            "firstdate_begin",
            "lastdate_end",
            "location_name",
            "location_address",
            "location_city",
            "location_district",
            "location_postalcode",
            "location_department",
            "location_region",
            "location_countrycode",
            "country_fr",
            "location_coordinates.lon",
            "location_coordinates.lat",
        ]
        df = df.reindex(columns=relevant_columns)
        df = df.drop_duplicates(subset=["uid"])
        records = df.to_dict(orient="records")
        return records
    except requests.RequestException as exc:
        raise RuntimeError(f"Erreur de connexion à Open Agenda : {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Réponse invalide reçue d'Open Agenda : {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de la récupération des événements : {exc}") from exc
    

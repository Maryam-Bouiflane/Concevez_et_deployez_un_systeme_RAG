from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class EventSearchFilters(BaseModel):
    """Filtres metadata extraits de la question utilisateur."""

    date_from: date | None = Field(
        default=None,
        description=(
            "Date de début de la période recherchée. "
            "Format YYYY-MM-DD."
        ),
    )

    date_to: date | None = Field(
        default=None,
        description=(
            "Date de fin de la période recherchée. "
            "Format YYYY-MM-DD."
        ),
    )

    location_city: str | None = Field(
        default=None,
        description="Ville explicitement indiquée par l'utilisateur.",
    )

    location_district: str | None = Field(
        default=None,
        description=(
            "Quartier ou arrondissement explicitement indiqué."
        ),
    )

    location_postalcode: str | None = Field(
        default=None,
        description="Code postal explicitement indiqué.",
    )

    location_department: str | None = Field(
        default=None,
        description="Département explicitement indiqué.",
    )

    location_region: str | None = Field(
        default=None,
        description="Région explicitement indiquée.",
    )

    location_countrycode: str | None = Field(
        default=None,
        description=(
            "Code pays ISO explicitement indiqué, par exemple FR."
        ),
    )

    country_fr: str | None = Field(
        default=None,
        description="Pays explicitement indiqué en français.",
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> "EventSearchFilters":
        """Vérifie que la période est cohérente."""

        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError(
                "date_from doit être antérieure ou égale à date_to."
            )

        return self


class EventSearchQuery(BaseModel):
    """
    Sortie structurée du Query Parser.

    Le LLM extrait uniquement les filtres metadata.
    La question originale de l'utilisateur est utilisée
    directement comme requête sémantique pour FAISS.
    """

    filters: EventSearchFilters | None = Field(
        default=None,
        description=(
            "Filtres metadata à appliquer aux événements. "
            "null si la question ne contient aucune contrainte "
            "temporelle ou géographique."
        ),
    )
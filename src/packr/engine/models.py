"""Scenario: the engine's input, validated on construction."""
from __future__ import annotations

import math
from dataclasses import dataclass

STORAGE = ("Ambient", "Chilled", "Frozen")
TRANSPORT = ("Normal", "Harsh")
LEVEL_NAMES = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
RESPIRATION_LEVELS = {name: level for level, name in LEVEL_NAMES.items()}

_ALIASES = {"moisture": "moisture_pct", "fat": "fat_pct", "rh": "rh_pct"}


def respiration_level_from_rate(mg_co2_kg_h: float) -> int:
    """Map a respiration rate to the engine's 0-3 scale (bins follow the seed table's AH-66 classes)."""
    if mg_co2_kg_h < 0:
        raise ValueError("respiration rate cannot be negative")
    if mg_co2_kg_h <= 5:
        return 0
    if mg_co2_kg_h <= 10:
        return 1
    if mg_co2_kg_h <= 20:
        return 2
    return 3


@dataclass(frozen=True)
class Scenario:
    commodity: str
    moisture_pct: float
    fat_pct: float
    ph: float
    shelf_life_days: float
    temp_c: float
    rh_pct: float
    storage: str = "Ambient"
    transport: str = "Normal"
    respiration: str = "None"                              # None / Low / Medium / High
    respiration_mgco2_kg_h: float | None = None            # measured AT the storage temperature; overrides `respiration`

    def __post_init__(self) -> None:
        def need(ok: bool, message: str) -> None:
            if not ok:
                raise ValueError(message)

        for name in ("moisture_pct", "fat_pct", "ph", "shelf_life_days", "temp_c", "rh_pct"):
            need(math.isfinite(getattr(self, name)), f"{name} must be a finite number")
        need(0 <= self.moisture_pct <= 100, "moisture_pct must be between 0 and 100")
        need(0 <= self.fat_pct <= 100, "fat_pct must be between 0 and 100")
        need(self.moisture_pct + self.fat_pct <= 100, "moisture_pct + fat_pct cannot exceed 100")
        need(0 <= self.ph <= 14, "ph must be between 0 and 14")
        need(self.shelf_life_days > 0, "shelf_life_days must be positive")
        need(-60 <= self.temp_c <= 60, "temp_c must be between -60 and 60")
        need(0 <= self.rh_pct <= 100, "rh_pct must be between 0 and 100")
        need(self.storage in STORAGE, f"storage must be one of {STORAGE}")
        need(self.transport in TRANSPORT, f"transport must be one of {TRANSPORT}")
        need(self.respiration in RESPIRATION_LEVELS, f"respiration must be one of {tuple(RESPIRATION_LEVELS)}")
        if self.respiration_mgco2_kg_h is not None:
            need(self.respiration_mgco2_kg_h >= 0, "respiration_mgco2_kg_h cannot be negative")

    @property
    def respiration_level(self) -> int:
        if self.respiration_mgco2_kg_h is not None:
            return respiration_level_from_rate(self.respiration_mgco2_kg_h)
        return RESPIRATION_LEVELS[self.respiration]

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Accept the short keys used in the demo notebook (moisture, fat, rh) as well as the full names."""
        return cls(**{_ALIASES.get(k, k): v for k, v in data.items()})

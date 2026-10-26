"""Rules: deterioration risks -> required packaging properties.

THE THRESHOLDS BELOW ARE UNVALIDATED STARTING POINTS, not literature values. They are the first
thing to calibrate (see ROADMAP.md, phase 3). Edit the tables here; nothing else hard-codes them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .models import LEVEL_NAMES, Scenario
from .risks import RiskAssessment

INF = math.inf
SHORT_LIFE_MAX_DAYS = 90

# index 0 = short shelf life (<= 90 days), index 1 = long shelf life (> 90 days)
OTR_MAX_BY_OXIDATION = {0: (INF, INF), 1: (1000, 200), 2: (200, 50), 3: (50, 5)}          # cm3/m2/day
WVTR_MAX_BY_MOISTURE_GAIN = {0: (INF, INF), 1: (50, 20), 2: (20, 5), 3: (5, 1)}           # g/m2/day
OTR_MIN_BY_RESPIRATION = {0: 0, 1: 0, 2: 500, 3: 2000}                                    # cm3/m2/day
DEHYDRATION_WVTR_MAX = 30


@dataclass(frozen=True)
class Requirements:
    otr_max: float
    otr_min: float
    wvtr_max: float
    light_min: float
    seal_min: int
    strength_min: int
    map_advice: str
    prefer_breathable: bool
    trace: tuple[str, ...]


def derive_requirements(s: Scenario, risks: RiskAssessment) -> Requirements:
    lv = risks.levels
    i = 1 if s.shelf_life_days > SHORT_LIFE_MAX_DAYS else 0
    life = f"long (>{SHORT_LIFE_MAX_DAYS} d)" if i else f"short (<={SHORT_LIFE_MAX_DAYS} d)"
    trace: list[str] = []

    otr_max = OTR_MAX_BY_OXIDATION[lv["oxidation"]][i]
    if otr_max < INF:
        trace.append(f"Oxidation {LEVEL_NAMES[lv['oxidation']]} + {life} shelf life -> OTR <= {otr_max:g}")

    otr_min = OTR_MIN_BY_RESPIRATION[lv["respiration"]]
    if otr_min:
        trace.append(f"Respiration {LEVEL_NAMES[lv['respiration']]} -> OTR >= {otr_min} (avoid anaerobic conditions)")

    wvtr_max = WVTR_MAX_BY_MOISTURE_GAIN[lv["moisture_gain"]][i]
    if wvtr_max < INF:
        trace.append(f"Moisture gain {LEVEL_NAMES[lv['moisture_gain']]} + {life} shelf life -> WVTR <= {wvtr_max:g}")
    if lv["moisture_loss"]:
        wvtr_max = min(wvtr_max, DEHYDRATION_WVTR_MAX)
        trace.append(f"Dehydration risk -> WVTR <= {DEHYDRATION_WVTR_MAX}")

    light_min = 0.8 if lv["light"] >= 2 else 0.0
    seal_min = 4 if (lv["oxidation"] >= 2 or lv["moisture_gain"] >= 2) else 3
    strength_min = 4 if s.transport == "Harsh" else 3
    trace.append(f"Transport {s.transport} -> mechanical strength >= {strength_min}/5")

    if lv["respiration"] >= 2:
        map_advice = "Recommended: passive MAP (indicative target O2 5-10%, CO2 10-20%, verify per commodity)"
    elif lv["oxidation"] == 3 and s.moisture_pct < 10:
        map_advice = "Recommended: nitrogen flushing to cut residual oxygen"
    elif lv["oxidation"] == 2:
        map_advice = "Optional"
    else:
        map_advice = "Not required"

    prefer_breathable = lv["respiration"] >= 3
    if prefer_breathable:
        trace.append("High respiration -> prefer micro-perforated / breathable film")

    return Requirements(otr_max, otr_min, wvtr_max, light_min, seal_min, strength_min,
                        map_advice, prefer_breathable, tuple(trace))

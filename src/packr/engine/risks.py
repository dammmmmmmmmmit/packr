"""Level 1: food deterioration risks, each scored 0 (none) to 3 (high) with a reason."""
from __future__ import annotations

from dataclasses import dataclass

from .models import LEVEL_NAMES, Scenario


@dataclass(frozen=True)
class RiskAssessment:
    levels: dict[str, int]
    reasons: tuple[str, ...]


def assess_risks(s: Scenario) -> RiskAssessment:
    levels: dict[str, int] = {}
    why: list[str] = []

    # Oxidation: driven by fat content, tightened in warm storage
    f = s.fat_pct
    ox = 3 if f >= 20 else 2 if f >= 10 else 1 if f >= 3 else 0
    if ox and s.temp_c >= 30:
        ox = min(3, ox + 1)
        why.append(f"Storage at {s.temp_c:g} C accelerates oxidation, risk raised one level")
    levels["oxidation"] = ox
    why.append(f"Fat {f:g}% -> oxidation risk {LEVEL_NAMES[ox]}")

    # Moisture gain (dry, crisp foods go soft) and moisture loss (fresh foods dehydrate)
    if s.moisture_pct < 10:
        gain = 3 if s.rh_pct >= 60 else 2 if s.rh_pct >= 40 else 1
        why.append(f"Moisture {s.moisture_pct:g}% with RH {s.rh_pct:g}% -> moisture gain risk {LEVEL_NAMES[gain]}")
    else:
        gain = 0
    levels["moisture_gain"] = gain
    loss = 2 if s.moisture_pct >= 70 else 0
    levels["moisture_loss"] = loss
    if loss:
        why.append(f"Moisture {s.moisture_pct:g}% -> dehydration risk {LEVEL_NAMES[loss]}")

    # Microbial: high moisture and non-acidic is the worst case
    if s.moisture_pct > 40 and s.ph > 4.6:
        micro = 3
    elif s.moisture_pct > 40:
        micro = 1
    else:
        micro = 0
    levels["microbial"] = micro
    if micro:
        why.append(f"Moisture {s.moisture_pct:g}% and pH {s.ph:g} -> microbial risk {LEVEL_NAMES[micro]}")

    # Light: photo-oxidation of fats
    levels["light"] = 2 if ox >= 2 else 0
    if levels["light"]:
        why.append("Fat-rich product is light sensitive -> needs a light barrier")

    # Respiration (fresh produce)
    levels["respiration"] = s.respiration_level
    if levels["respiration"]:
        why.append(f"Respiration {LEVEL_NAMES[levels['respiration']]} -> film must let O2/CO2 exchange")

    return RiskAssessment(levels, tuple(why))

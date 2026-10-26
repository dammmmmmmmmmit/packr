"""Recommendation: scenario in, ranked packaging structures plus the reasoning out."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from ..materials import load_materials
from .models import Scenario
from .risks import RiskAssessment, assess_risks
from .rules import Requirements, derive_requirements
from .structures import Structure, build_candidates, check, evaluate


@dataclass(frozen=True)
class Recommendation:
    scenario: Scenario
    risks: RiskAssessment
    requirements: Requirements
    top: tuple[Structure, ...]
    n_scanned: int
    n_passing: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-safe dict (infinite limits become None). This is the shape the ML stage will train on."""
        return _clean(asdict(self))


def _clean(value):
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def recommend(s: Scenario, materials: pd.DataFrame | None = None, top_k: int = 3) -> Recommendation:
    materials = load_materials() if materials is None else materials
    risks = assess_risks(s)
    req = derive_requirements(s, risks)

    scored = [evaluate(c, materials) for c in build_candidates(materials)]
    passing = [st for st in scored if not check(st, req)]
    passing.sort(key=lambda st: ((0 if st.breathable else 1) if req.prefer_breathable else 0,
                                 st.cost, st.n, st.WVTR))

    notes: list[str] = []
    if (materials["trust"] == "placeholder").any():
        notes.append("Material barrier values are placeholders, not datasheet or literature values.")
    if s.storage != "Ambient":
        notes.append(f"Storage type '{s.storage}' is recorded but the rules currently use only temperature and RH.")
    if s.respiration_mgco2_kg_h is not None:
        notes.append("Respiration rate is taken as measured at the storage temperature "
                     "(use packr.units.q10_adjust if it was measured at another temperature).")
    if not passing:
        notes.append("No candidate structure meets all requirements: relax a rule or add materials.")

    return Recommendation(s, risks, req, tuple(passing[:top_k]), len(scored), len(passing), tuple(notes))

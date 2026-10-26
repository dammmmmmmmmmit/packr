"""Plain-text report for a Recommendation."""
from __future__ import annotations

import math

from .recommend import Recommendation


def _band(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def _otr_text(req) -> str:
    low, high = req.otr_min, req.otr_max
    if math.isinf(high) and not low:
        return "Not restricted"
    if math.isinf(high):
        return f"At least {low:g} cc/m2/day (breathable)"
    text = f"{_band(high, (1, 10, 100), ('Very Low', 'Low', 'Medium', 'High'))} (<= {high:g} cc/m2/day)"
    return text + (f", and >= {low:g} for respiration" if low else "")


def _wvtr_text(req) -> str:
    if math.isinf(req.wvtr_max):
        return "Not restricted"
    band = _band(req.wvtr_max, (1, 5, 20), ("Very Low", "Low", "Medium", "High"))
    return f"{band} (<= {req.wvtr_max:g} g/m2/day)"


def format_report(rec: Recommendation) -> str:
    s, req = rec.scenario, rec.requirements
    out = ["=" * 64,
           f"INPUT: {s.commodity} | moisture {s.moisture_pct:g}% | fat {s.fat_pct:g}% | pH {s.ph:g} | "
           f"respiration {s.respiration if s.respiration_mgco2_kg_h is None else f'{s.respiration_mgco2_kg_h:g} mg CO2/kg/h'}",
           f"       shelf life {s.shelf_life_days:g} d | {s.temp_c:g} C | RH {s.rh_pct:g}% | {s.storage} | transport {s.transport}",
           "", "WHY (Level 1 risks):", *[f"  * {r}" for r in rec.risks.reasons],
           "", "WHY (requirement rules):", *[f"  * {r}" for r in req.trace]]
    if not rec.top:
        out += ["", *[f"NOTE: {n}" for n in rec.notes]]
        return "\n".join(out)

    best = rec.top[0]
    out += ["", f"RECOMMENDED PACKAGING: {best.n} layer(s)   ({rec.n_passing} of {rec.n_scanned} candidates passed)"]
    for i, name in enumerate(best.layers, 1):
        role = "monolayer" if best.n == 1 else "outer" if i == 1 else "sealant" if i == best.n else "barrier"
        out.append(f"  layer {i} ({role}): {name}")
    out += ["", "REQUIRED PROPERTIES:",
            f"  OTR                 -> {_otr_text(req)}",
            f"  WVTR                -> {_wvtr_text(req)}",
            f"  Light barrier       -> {'High' if req.light_min >= 0.8 else 'Not critical'}",
            f"  Sealability         -> {'High' if req.seal_min >= 4 else 'Moderate'}",
            f"  Mechanical strength -> {'Moderate/High' if req.strength_min >= 4 else 'Moderate'}",
            f"  MAP                 -> {req.map_advice}",
            "", "ACHIEVED BY THIS STRUCTURE:",
            f"  OTR {best.OTR:.3g} cc/m2/day | WVTR {best.WVTR:.3g} g/m2/day | light {best.light:.2f} | "
            f"seal {best.seal:g}/5 | strength {best.strength:.1f}/5 | ~{best.thickness_um:g} um | "
            f"recyclable: {'yes' if best.recyclable else 'no'}",
            "", "TOP ALTERNATIVES (ranked by cost, then layers, then WVTR):"]
    for st in rec.top:
        out.append(f"  {' / '.join(st.layers):<28} OTR {st.OTR:<9.3g} WVTR {st.WVTR:<8.3g} "
                   f"cost {st.cost:g}  {st.thickness_um:g} um  recyclable={st.recyclable}")
    out += ["", *[f"NOTE: {n}" for n in rec.notes]]
    return "\n".join(out)

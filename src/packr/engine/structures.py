"""Level 4: candidate structures (monolayer and laminates) and their combined properties.

Layer order is outer -> (optional barrier layer) -> sealant. Laminate OTR/WVTR combine in series,
light barrier is the best single layer, strength is the layer average, cost and thickness are summed.
A structure counts as recyclable only if it is a single polymer family (a deliberately simple first rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..units import series_transmission
from .rules import Requirements


@dataclass(frozen=True)
class Structure:
    layers: tuple[str, ...]
    OTR: float
    WVTR: float
    light: float
    seal: float
    strength: float
    cost: float
    thickness_um: float
    recyclable: bool
    breathable: bool

    @property
    def n(self) -> int:
        return len(self.layers)


def _with_role(materials: pd.DataFrame, role: str) -> list[str]:
    return [name for name, row in materials.iterrows() if role in row["roles"]]


def build_candidates(materials: pd.DataFrame) -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = [(m,) for m in _with_role(materials, "mono")]
    for outer in _with_role(materials, "outer"):
        for mid in [None, *_with_role(materials, "mid")]:
            if mid == outer:
                continue
            for sealant in _with_role(materials, "sealant"):
                candidates.append((outer, *((mid,) if mid else ()), sealant))
    return candidates


def evaluate(layers: tuple[str, ...], materials: pd.DataFrame) -> Structure:
    rows = [materials.loc[name] for name in layers]
    return Structure(
        layers=tuple(layers),
        OTR=series_transmission([float(r["OTR"]) for r in rows]),
        WVTR=series_transmission([float(r["WVTR"]) for r in rows]),
        light=max(float(r["light"]) for r in rows),
        seal=float(rows[-1]["seal"]),                       # the innermost layer does the sealing
        strength=sum(float(r["strength"]) for r in rows) / len(rows),
        cost=sum(float(r["cost"]) for r in rows),
        thickness_um=sum(float(r["gauge_um"]) for r in rows),
        recyclable=len({r["family"] for r in rows}) == 1,
        breathable=any(bool(r["breathable"]) for r in rows),
    )


def check(st: Structure, req: Requirements) -> list[str]:
    """Return the reasons a structure fails the requirements (empty list = passes)."""
    fails: list[str] = []
    if st.OTR > req.otr_max:
        fails.append(f"OTR {st.OTR:.3g} is above the limit {req.otr_max:g}")
    if st.OTR < req.otr_min:
        fails.append(f"OTR {st.OTR:.3g} is below the minimum {req.otr_min:g}")
    if st.WVTR > req.wvtr_max:
        fails.append(f"WVTR {st.WVTR:.3g} is above the limit {req.wvtr_max:g}")
    if st.light < req.light_min:
        fails.append(f"light barrier {st.light:.2f} is below {req.light_min:g}")
    if st.seal < req.seal_min:
        fails.append(f"sealability {st.seal:g} is below {req.seal_min}")
    if st.strength < req.strength_min:
        fails.append(f"strength {st.strength:.1f} is below {req.strength_min}")
    return fails

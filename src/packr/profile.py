"""Commodity profile: the commodity-specific simulator parameters the plan's ten inputs do not determine (spec section 6 and 7.2).

Source order for one request:
  1. user-entered values (`overrides`),
  2. the commodity table row, if the name is in the table,
  3. otherwise the mean of the 3 nearest table commodities in the same category (standardised moisture, fat, log respiration,
     log base life; a dimension is used only if the query has a value for it and it varies within the category).

`source` is "table", "user" (at least one override) or "estimated" (at least one value came from neighbours). An estimated
profile is a guess: callers must flag it and widen any interval they report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .commodities import load_commodities
from .sim.physics import COMMODITY_COLS

CATEGORIES = ("fresh_produce", "dry_snack")
# everything the simulator needs about a commodity except composition, which comes from the request
PROFILE_COLS = [c for c in COMMODITY_COLS if c not in ("moisture_pct", "fat_pct")]
SNACK_MAX_MOISTURE_PCT = 15.0   # used only to guess the category of a food that is not in the table and has no category given
_NN_K = 3
_BOOL_COLS = ("chilling_sensitive",)


@dataclass(frozen=True)
class Profile:
    commodity: str                  # table name if matched, else the name as given
    category: str
    values: dict                    # PROFILE_COLS -> float (NaN where a parameter does not apply to the category)
    source: str                     # table / user / estimated
    in_table: bool
    neighbours: tuple[str, ...] = ()
    estimated_fields: tuple[str, ...] = ()
    user_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())


def find_commodity(name: str, master: pd.DataFrame) -> pd.Series | None:
    """Case-insensitive match on the table name, also trying a simple singular ("Strawberries" -> "Strawberry")."""
    key = name.strip().lower()
    names = master["commodity"].str.lower()
    tries = [key]
    if key.endswith("ies"):
        tries.append(key[:-3] + "y")
    if key.endswith("es"):
        tries.append(key[:-2])
    if key.endswith("s"):
        tries.append(key[:-1])
    for k in tries:
        hit = master[names == k]
        if len(hit):
            return hit.iloc[0]
    return None


def infer_category(moisture_pct: float) -> str:
    return "dry_snack" if moisture_pct <= SNACK_MAX_MOISTURE_PCT else "fresh_produce"


def nearest(master: pd.DataFrame, category: str, moisture_pct: float, fat_pct: float, resp_ref: float | None = None,
            base_life_days: float | None = None, k: int = _NN_K) -> pd.DataFrame:
    """The k table commodities of `category` closest to the query in standardised property space."""
    pool = master[master["category"] == category]
    if pool.empty:
        raise ValueError(f"no commodities of category {category!r} in the table")
    dims = {"moisture_pct": (pool["moisture_pct"], moisture_pct), "fat_pct": (pool["fat_pct"], fat_pct)}
    if resp_ref is not None:
        dims["log_resp"] = (np.log1p(pool["resp_ref_mgco2_kg_h"]), np.log1p(resp_ref))
    if base_life_days is not None:
        dims["log_life"] = (np.log(pool["base_life_days"]), np.log(base_life_days))
    d2 = np.zeros(len(pool))
    for col, q in dims.values():
        sd = float(col.std())
        if sd > 0 and np.isfinite(sd):
            d2 += ((col.to_numpy(dtype=float) - q) / sd) ** 2
    return pool.iloc[np.argsort(d2, kind="stable")[:k]]


def _mean_profile(rows: pd.DataFrame) -> dict:
    out = {}
    for c in PROFILE_COLS:
        v = rows[c].astype(float)
        out[c] = float(v.mean() >= 0.5) if c in _BOOL_COLS else (float(v.mean()) if v.notna().any() else np.nan)
    return out


def resolve_profile(name: str, moisture_pct: float, fat_pct: float, category: str | None = None, overrides: dict | None = None,
                    resp_ref: float | None = None, master: pd.DataFrame | None = None) -> Profile:
    master = load_commodities() if master is None else master
    overrides = {k: float(v) for k, v in (overrides or {}).items() if v is not None}
    unknown = set(overrides) - set(PROFILE_COLS)
    if unknown:
        raise ValueError(f"unknown commodity_profile fields {sorted(unknown)}; allowed: {PROFILE_COLS}")
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    notes: list[str] = []
    row = find_commodity(name, master)
    if row is not None:
        if category is not None and category != row["category"]:
            raise ValueError(f"{row['commodity']} is a {row['category']} in the commodity table, not {category}")
        values = {c: float(row[c]) for c in PROFILE_COLS}
        values.update(overrides)
        return Profile(str(row["commodity"]), str(row["category"]), values, "user" if overrides else "table", True,
                       user_fields=tuple(sorted(overrides)))

    if category is None:
        category = infer_category(moisture_pct)
        notes.append(f"'{name}' is not in the commodity table; category guessed as {category} from moisture {moisture_pct:g}%.")
    nn = nearest(master, category, moisture_pct, fat_pct, resp_ref=overrides.get("resp_ref_mgco2_kg_h", resp_ref),
                 base_life_days=overrides.get("base_life_days"))
    values = _mean_profile(nn)
    values.update(overrides)
    estimated = tuple(c for c in PROFILE_COLS if c not in overrides and not (np.isnan(values[c]) and nn[c].isna().all()))
    source = "estimated" if estimated else "user"
    if estimated:
        notes.append(f"Profile for '{name}' estimated from its nearest table commodities ({', '.join(nn['commodity'])}); "
                     "treat the success rate as a rough guess.")
    return Profile(name, category, values, source, False, tuple(nn["commodity"]), estimated, tuple(sorted(overrides)), tuple(notes))

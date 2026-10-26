"""Packaging material table.

Each material is either
  * "bulk"  : a homogeneous polymer. Barrier is stored as an intrinsic permeability coefficient
              (p_o2, p_wv), so OTR/WVTR scale as 1/thickness.
  * "fixed" : a coated, metallised, perforated or foil layer whose barrier is set by the coating,
              pinholes or perforations rather than the substrate thickness (otr_fixed, wvtr_fixed).

The shipped table is a PLACEHOLDER (trust == "placeholder"). Replace rows with values from
the TRANSMAT-derived tables or supplier datasheets and set trust accordingly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import REFERENCE_DIR
from .units import WVTR_DP_KPA

MATERIALS_CSV = REFERENCE_DIR / "materials_placeholder.csv"

REQUIRED = {"name", "family", "roles", "breathable", "barrier_type", "gauge_um", "p_o2", "p_wv",
            "otr_fixed", "wvtr_fixed", "light", "seal", "strength", "cost", "source", "trust"}
ROLES = {"outer", "mid", "sealant", "mono"}


def load_materials(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the material table. Adds OTR and WVTR at the nominal gauge."""
    df = pd.read_csv(path or MATERIALS_CSV)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"materials table is missing columns: {sorted(missing)}")
    if df["name"].duplicated().any():
        raise ValueError(f"duplicate material names: {df.loc[df['name'].duplicated(), 'name'].tolist()}")
    df["roles"] = df["roles"].map(lambda s: [r.strip() for r in str(s).split(";") if r.strip()])
    unknown = {r for roles in df["roles"] for r in roles} - ROLES
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; allowed: {sorted(ROLES)}")
    if not df["barrier_type"].isin(["bulk", "fixed"]).all():
        raise ValueError("barrier_type must be 'bulk' or 'fixed'")
    df["breathable"] = df["breathable"].astype(bool)

    bulk = (df["barrier_type"] == "bulk").to_numpy()
    t_mm = df["gauge_um"].to_numpy(dtype=float) / 1000.0
    df["OTR"] = np.where(bulk, df["p_o2"] / t_mm, df["otr_fixed"])
    df["WVTR"] = np.where(bulk, df["p_wv"] * WVTR_DP_KPA / t_mm, df["wvtr_fixed"])
    if df[["OTR", "WVTR"]].isna().any().any():
        bad = df.loc[df[["OTR", "WVTR"]].isna().any(axis=1), "name"].tolist()
        raise ValueError(f"materials without usable barrier data: {bad}")
    return df.set_index("name")


RH_FACTORS_CSV = REFERENCE_DIR / "rh_permeability.csv"


def load_rh_factors(path: str | Path | None = None) -> pd.DataFrame:
    """Sourced OTR-vs-RH curves (material, rh_pct, otr_multiplier relative to the table OTR). Empty = not sourced."""
    df = pd.read_csv(path or RH_FACTORS_CSV)
    return df.dropna(subset=["material", "rh_pct", "otr_multiplier"])


def layer_rh(layers: list[str], rh_in: float, rh_out: float, materials: pd.DataFrame, gauge_scale: float = 1.0) -> list[float]:
    """RH (%) at the middle of each layer, outer -> inner, by linear interpolation in cumulative water-vapour resistance
    (1/WVTR of each layer) from the outside RH to the inside RH: a steady-state approximation of the gradient."""
    res = []
    for name in layers:
        row = materials.loc[name]
        w = float(row["WVTR"]) / (gauge_scale if row["barrier_type"] == "bulk" else 1.0)
        res.append(1.0 / w)
    total = sum(res)
    out, acc = [], 0.0
    for r in res:
        out.append(rh_out + (rh_in - rh_out) * (acc + r / 2.0) / total)
        acc += r
    return out


def otr_rh_multiplier(layers: list[str], rh_in: float, rh_out: float, materials: pd.DataFrame, table: pd.DataFrame,
                      gauge_scale: float = 1.0) -> float:
    """OTR of the laminate at the RH each layer sees, divided by its table OTR. Layers without a curve keep multiplier 1;
    a curve is interpolated linearly in RH and clamped at its ends."""
    rhs = layer_rh(layers, rh_in, rh_out, materials, gauge_scale)
    inv_nom, inv_rh = 0.0, 0.0
    for name, rh in zip(layers, rhs):
        row = materials.loc[name]
        otr = float(row["OTR"]) / (gauge_scale if row["barrier_type"] == "bulk" else 1.0)
        curve = table[table["material"] == name].sort_values("rh_pct")
        mult = float(np.interp(rh, curve["rh_pct"], curve["otr_multiplier"])) if len(curve) else 1.0
        inv_nom += 1.0 / otr
        inv_rh += 1.0 / (otr * mult)
    return inv_nom / inv_rh


def required_thickness_um(materials: pd.DataFrame, name: str, otr_max: float) -> float:
    """Gauge (um) at which a bulk polymer's OTR equals the limit: the mathematical minimum thickness.

    Real films have a minimum practical gauge and a maximum sensible one; the table does not hold those
    yet (roadmap phase 2), so a result far below or above the nominal gauge is a signal, not a spec.
    Fixed-barrier layers cannot be tuned this way.
    """
    row = materials.loc[name]
    if row["barrier_type"] != "bulk":
        raise ValueError(f"{name} is a fixed-barrier layer; its OTR does not scale with thickness")
    return 1000.0 * float(row["p_o2"]) / otr_max

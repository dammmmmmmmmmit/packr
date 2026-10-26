"""Model features. Shared by the sufficiency study, the phase 5 model and the optimiser.

FEATURES are the plan inputs plus the package spec. LATENT is the commodity profile (section 7.2 of the spec): it is not in
the plan's ten inputs, so at prediction time it is looked up from the commodity table, typed in by the user, or estimated
from the nearest commodities (see `packr.profile`). Missing values (a snack has no `o2_min_pct`) stay NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["is_snack", "moisture_pct", "fat_pct", "ph", "resp_rate_mgco2_kg_h", "log_target_days", "temp_c", "rh_pct", "chilled",
            "harsh", "log_otr", "log_wvtr", "light_barrier", "seal", "strength", "thickness_um", "n_layers", "breathable",
            "mass_kg", "area_m2_per_kg", "log_perf", "flush_o2_pct"]
LATENT = ["base_life_days", "q10_decay", "o2_min_pct", "co2_max_pct", "crit_weight_loss_pct", "ma_gain", "transp_g_kg_d",
          "crit_aw", "iso_b", "ox_ref_days", "ea_ox_kj", "light_sens", "ideal_temp_c", "fragility"]
LIFE_FLOOR_DAYS = 0.05          # shelf-life labels are logged; a simulated life of 0 (already past critical moisture) is clipped here


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["is_snack"] = (df["category"] == "dry_snack").astype(int)
    for c in ("moisture_pct", "fat_pct", "ph", "resp_rate_mgco2_kg_h", "temp_c", "rh_pct", "harsh", "light_barrier", "seal", "strength",
              "thickness_um", "n_layers", "breathable", "mass_kg", "area_m2_per_kg", "flush_o2_pct"):
        x[c] = df[c]
    x["log_target_days"] = np.log(df["target_days"])
    x["chilled"] = (df["storage"] == "Chilled").astype(int)
    x["log_otr"] = np.log10(df["otr23"])
    x["log_wvtr"] = np.log10(df["wvtr38"])
    x["log_perf"] = np.log1p(df["perf_mm2_per_kg"])
    return x[FEATURES]


def featurise_with_profile(df: pd.DataFrame) -> pd.DataFrame:
    """FEATURES followed by the LATENT commodity profile, as floats (NaN where a parameter does not apply)."""
    return pd.concat([featurise(df), df[LATENT].astype(float)], axis=1)

"""Commodity master table: everything the simulator needs to know about one food.

Built from three inputs, each with a stated trust level:

  * the 27-row fresh-produce seed (respiration, ideal storage, typical life)       trust: unverified_seed
  * the 25-food supplied CSV pair (moisture, pH, class labels, observed lives)       trust: supplied_csv
  * the numbers typed in below (simulator parameters, composition for foods missing
    from the sources, the dry-snack rows)                                            trust: assumed

EVERY "assumed" number is a literature-informed order of magnitude written from general knowledge, not
a looked-up value. They are the first things to verify (see docs/IMPLEMENTATION_SPEC.md, section 11).
Run `python -m packr.commodities` to rebuild reference/commodities_master.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import REFERENCE_DIR

MASTER_CSV = REFERENCE_DIR / "commodities_master.csv"
SEED_CSV = REFERENCE_DIR / "commodities_produce_seed.csv"
SUPPLIED_CLASSIFIER = REFERENCE_DIR / "supplied_classifier.csv"
SUPPLIED_SHELF_LIFE = REFERENCE_DIR / "supplied_shelf_life.csv"
COMPOSITION_FDC_CSV = REFERENCE_DIR / "composition_fdc.csv"      # moisture and fat from USDA FoodData Central (packr.fdc)

RESP_REF_TEMP_C = 10.0      # ASSUMPTION: the seed does not record a reference temperature
# mg CO2/kg/h at 10 C for a class label; midpoints of the seed values in each class
RESP_CLASS_VALUE = {"Very Low": 5.0, "Low": 9.0, "Medium": 17.0, "High": 27.0, "Very High": 62.0}
FRAGILITY_VALUE = {"Low": 0.6, "Medium": 1.0, "High": 1.5}

# name -> (group, o2_min_pct, co2_max_pct, critical_weight_loss_pct, ma_gain, transpiration g/kg/day per unit RH at 20 C,
#          fragility, q10_decay). ASSUMED values, see module docstring.
PRODUCE_PARAMS: dict[str, tuple] = {
    "Strawberry": ("berry", 2, 15, 6, 1.0, 40, 1.5, 2.6), "Blueberry": ("berry", 2, 15, 6, 0.8, 30, 1.5, 2.4),
    "Grapes": ("berry", 1, 12, 5, 0.6, 25, 1.5, 2.2),
    "Lettuce": ("leafy", 1, 3, 5, 0.9, 100, 1.0, 2.5), "Spinach": ("leafy", 1, 10, 5, 0.9, 110, 1.0, 2.6),
    "Cabbage": ("leafy", 2, 8, 8, 0.5, 30, 0.6, 2.0), "Asparagus": ("leafy", 1, 12, 6, 0.8, 70, 1.0, 2.6),
    "Broccoli": ("crucifer", 1, 10, 5, 0.9, 90, 1.0, 2.6), "Cauliflower": ("crucifer", 2, 5, 6, 0.7, 60, 1.0, 2.4),
    "Mushroom": ("fungi", 2, 15, 5, 0.8, 90, 1.0, 2.6),
    "Banana": ("fruit", 2, 5, 7, 0.7, 12, 1.0, 2.4), "Tomato": ("fruit", 2, 5, 6, 0.7, 15, 1.0, 2.3),
    "Mango": ("fruit", 2, 8, 8, 0.6, 15, 1.0, 2.4), "Papaya": ("fruit", 2, 8, 8, 0.6, 15, 1.0, 2.4),
    "Guava": ("fruit", 2, 8, 8, 0.6, 18, 1.0, 2.3), "Avocado": ("fruit", 2, 10, 8, 0.6, 10, 1.0, 2.4),
    "Peach": ("fruit", 1, 5, 6, 0.6, 25, 1.2, 2.3), "Pear": ("fruit", 1, 3, 6, 0.5, 12, 1.0, 2.0),
    "Apple": ("fruit", 1.5, 3, 7, 0.5, 10, 0.6, 2.0), "Pineapple": ("fruit", 2, 10, 8, 0.5, 8, 0.6, 2.2),
    "Orange": ("fruit", 5, 5, 10, 0.2, 8, 0.6, 2.0), "Pomegranate": ("fruit", 3, 10, 10, 0.3, 8, 0.6, 2.0),
    "Cucumber": ("fruit_veg", 2, 3, 5, 0.5, 40, 1.0, 2.4), "Bell Pepper": ("fruit_veg", 2, 5, 6, 0.5, 35, 1.0, 2.3),
    "Green Chilli": ("fruit_veg", 2, 5, 6, 0.5, 35, 0.6, 2.3), "Okra": ("fruit_veg", 2, 5, 6, 0.6, 45, 1.0, 2.5),
    "Brinjal": ("fruit_veg", 2, 5, 6, 0.5, 25, 1.0, 2.3), "Green Peas": ("fruit_veg", 2, 10, 6, 0.7, 60, 0.6, 2.6),
    "Sweet corn": ("fruit_veg", 2, 15, 6, 0.8, 50, 1.0, 2.6),
    "Potato": ("root", 2, 5, 10, 0.2, 6, 0.6, 1.8), "Sweet potato": ("root", 2, 5, 10, 0.2, 8, 0.6, 1.8),
    "Carrot": ("root", 2, 5, 8, 0.4, 35, 0.6, 2.0), "Beet": ("root", 2, 5, 8, 0.3, 20, 0.6, 2.0),
    "Onion": ("root", 1, 10, 10, 0.3, 5, 0.6, 1.8), "Garlic": ("root", 1, 10, 10, 0.3, 4, 0.6, 1.8),
    "Winter squash": ("root", 2, 5, 10, 0.2, 5, 0.6, 1.8),
}

# composition for produce that is in the seed but not in the supplied CSV (rounded handbook-style values, ASSUMED)
PRODUCE_COMPOSITION_FALLBACK = {   # name: (moisture_pct, pH)
    "Orange": (87, 3.7), "Pineapple": (86, 3.5), "Avocado": (73, 6.3), "Peach": (89, 3.7), "Pear": (84, 4.0),
    "Sweet potato": (77, 5.8), "Beet": (88, 5.8), "Winter squash": (88, 5.5), "Mushroom": (92, 6.3),
    "Sweet corn": (76, 6.5), "Asparagus": (93, 6.0),
}
SEED_NAME_ALIAS = {"Grape": "Grapes", "Bell pepper": "Bell Pepper"}

# dry snacks: ALL ASSUMED. crit_aw = water activity at which texture is lost; iso_b = slope of a linear moisture
# isotherm (g water / g solids per unit aw); ox_ref_days = days to rancidity at 25 C in air with some light.
SNACK_ROWS = [
    # name,               moist, fat, pH,  crit_aw, iso_b, ox_ref, Ea_ox, light_sens, fragility, base_life, ideal_T, ideal_RH
    ("Potato Chips",        2.0, 34.0, 6.0, 0.45, 0.070, 150, 65, 0.8, 1.5, 180, 20, 50),
    ("Tortilla Chips",      1.5, 25.0, 6.5, 0.45, 0.080, 200, 65, 0.6, 1.5, 180, 20, 50),
    ("Banana Chips",        3.0, 30.0, 5.0, 0.50, 0.090, 180, 65, 0.7, 1.5, 150, 20, 50),
    ("Extruded Puffs",      2.0, 28.0, 6.5, 0.45, 0.100, 220, 65, 0.6, 1.5, 180, 20, 50),
    ("Salted Crackers",     3.0, 12.0, 6.5, 0.50, 0.120, 300, 60, 0.4, 1.0, 270, 20, 50),
    ("Cream Biscuits",      3.5, 20.0, 6.5, 0.55, 0.130, 365, 60, 0.4, 1.0, 270, 20, 50),
    ("Namkeen Sev",         4.0, 30.0, 6.0, 0.50, 0.100, 120, 65, 0.7, 1.0, 120, 20, 50),
    ("Roasted Peanuts",     2.0, 48.0, 6.5, 0.60, 0.060, 180, 70, 0.7, 0.6, 240, 20, 50),
    ("Roasted Almonds",     3.5, 50.0, 6.5, 0.60, 0.060, 300, 70, 0.6, 0.6, 300, 20, 50),
    ("Ready Popcorn",       3.0, 20.0, 6.5, 0.50, 0.120, 150, 60, 0.6, 1.5, 120, 20, 50),
    ("Corn Flakes",         3.0,  1.0, 6.5, 0.45, 0.140, 700, 55, 0.2, 1.0, 365, 20, 50),
    ("Instant Noodles",     5.0, 17.0, 6.0, 0.60, 0.130, 270, 65, 0.4, 1.0, 270, 20, 50),
]


# frying / formulation oil of each snack, for the B2 oxidation model (reference/oil_stability.csv). name -> (oil, doi).
# EMPTY until sourced: an unsourced snack keeps oil_type empty and the simulator falls back to the v1 oxidation model.
OIL_TYPE: dict[str, tuple[str, str]] = {
    "Banana Chips": ("coconut", "10.1007/s13197-012-0839-0"),     # Manikantan et al. 2012: chips fried in coconut oil (approved by Adi)
}


def _supplied_summary() -> pd.DataFrame:
    cls = pd.read_csv(SUPPLIED_CLASSIFIER)
    sl = pd.read_csv(SUPPLIED_SHELF_LIFE)
    non_map = sl[sl["Packaging_Material_Type"] != "MAP High Barrier"]
    g = cls.groupby("Food_Type").agg(
        resp_class=("Respiration_Rate", "first"), ethylene_prod=("Ethylene_Production", "first"),
        ethylene_sens=("Ethylene_Sensitivity", "first"), moisture_pct=("Moisture_Content_Pct", "first"),
        chilling=("Chilling_Sensitivity", "first"), fragility_label=("Mechanical_Fragility", "first"))
    g["ph"] = sl.groupby("Food_Type")["pH_Level"].first()
    g["ideal_temp_c"] = sl.groupby("Food_Type")["Storage_Temperature_C"].median()
    g["ideal_rh_pct"] = sl.groupby("Food_Type")["Relative_Humidity_Pct"].median()
    g["base_life_days"] = non_map.groupby("Food_Type")["Shelf_Life_Days"].median()
    return g


def build_master() -> pd.DataFrame:
    seed = pd.read_csv(SEED_CSV)
    seed["commodity"] = seed["commodity"].replace(SEED_NAME_ALIAS)
    seed = seed.set_index("commodity")
    sup = _supplied_summary()
    rows = []
    for name in sorted(set(seed.index) | set(sup.index)):
        s = seed.loc[name] if name in seed.index else None
        c = sup.loc[name] if name in sup.index else None
        grp, o2min, co2max, wl, ma, tr, frag, q10d = PRODUCE_PARAMS[name]
        if s is not None:
            resp, resp_src = float(s["respiration_rate_mgco2_kg_hr"]), "seed"
        else:
            resp, resp_src = RESP_CLASS_VALUE[c["resp_class"]], "class_midpoint"
        moist = float(c["moisture_pct"]) if c is not None else PRODUCE_COMPOSITION_FALLBACK[name][0]
        ph = float(c["ph"]) if c is not None else PRODUCE_COMPOSITION_FALLBACK[name][1]
        chilling = bool(s["chilling_sensitive"]) if s is not None else (c["chilling"] == "Yes")
        rows.append(dict(
            commodity=name, category="fresh_produce", subgroup=grp, moisture_pct=moist, fat_pct=0.3, ph=ph,
            resp_ref_mgco2_kg_h=resp, resp_ref_temp_c=RESP_REF_TEMP_C, q10_resp=2.5,
            ideal_temp_c=float(s["ideal_temp_c"]) if s is not None else float(c["ideal_temp_c"]),
            ideal_rh_pct=float(s["ideal_rh_pct"]) if s is not None else float(c["ideal_rh_pct"]),
            base_life_days=float(s["typical_shelf_life_days"]) if s is not None else float(c["base_life_days"]),
            chilling_sensitive=chilling, fragility=FRAGILITY_VALUE[c["fragility_label"]] if c is not None else frag,
            o2_min_pct=o2min, co2_max_pct=co2max, crit_weight_loss_pct=wl, ma_gain=ma, transp_g_kg_d=tr, q10_decay=q10d,
            crit_aw=np.nan, iso_b=np.nan, ox_ref_days=np.nan, ea_ox_kj=np.nan, light_sens=np.nan,
            in_seed=s is not None, in_supplied_csv=c is not None,
            trust_composition="supplied_csv" if c is not None else "assumed",
            trust_respiration="unverified_seed" if s is not None else "assumed_class_midpoint",
            trust_sim_params="assumed"))
    for (name, m, f, ph, caw, b, ox, ea, ls, frag, life, tid, rh) in SNACK_ROWS:
        rows.append(dict(
            commodity=name, category="dry_snack", subgroup="snack", moisture_pct=m, fat_pct=f, ph=ph,
            resp_ref_mgco2_kg_h=0.0, resp_ref_temp_c=RESP_REF_TEMP_C, q10_resp=2.5, ideal_temp_c=tid, ideal_rh_pct=rh,
            base_life_days=life, chilling_sensitive=False, fragility=frag, o2_min_pct=np.nan, co2_max_pct=np.nan,
            crit_weight_loss_pct=np.nan, ma_gain=np.nan, transp_g_kg_d=np.nan, q10_decay=np.nan,
            crit_aw=caw, iso_b=b, ox_ref_days=ox, ea_ox_kj=ea, light_sens=ls,
            in_seed=False, in_supplied_csv=False, trust_composition="assumed", trust_respiration="n/a",
            trust_sim_params="assumed"))
    df = pd.DataFrame(rows)
    df.insert(0, "commodity_id", [f"C{i:03d}" for i in range(len(df))])
    df["oil_type"] = df["commodity"].map(lambda n: OIL_TYPE[n][0] if n in OIL_TYPE else np.nan)
    df["oil_type_doi"] = df["commodity"].map(lambda n: OIL_TYPE[n][1] if n in OIL_TYPE else np.nan)
    return apply_fdc_composition(df)


def apply_fdc_composition(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Overwrite moisture and fat with the USDA FoodData Central values in reference/composition_fdc.csv (built by
    `python -m packr.fdc`, approved 2026-10-26). pH is not in FDC, so trust_composition records both sources."""
    fdc = pd.read_csv(path or COMPOSITION_FDC_CSV).set_index("commodity")
    df = df.copy()
    for i, r in df.iterrows():
        if r["commodity"] not in fdc.index:
            continue
        f = fdc.loc[r["commodity"]]
        df.at[i, "moisture_pct"], df.at[i, "fat_pct"] = float(f["moisture_pct"]), float(f["fat_pct"])
        srcs = sorted({f["moisture_pct_source"], f["fat_pct_source"]})
        tag = " / ".join(s.replace("FDC ", "fdc_").lower().replace(" ", "_") for s in srcs)
        df.at[i, "trust_composition"] = f"moisture+fat: {tag}; ph: {r['trust_composition']}"
    return df


def load_commodities(path=None) -> pd.DataFrame:
    return pd.read_csv(path or MASTER_CSV)


if __name__ == "__main__":
    out = build_master()
    out.to_csv(MASTER_CSV, index=False)
    print(f"wrote {MASTER_CSV} ({len(out)} commodities: {out['category'].value_counts().to_dict()})")

"""Spec T2: moisture and fat for the commodity table from USDA FoodData Central (FDC).

    python -m packr.fdc --raw data/raw --out data/processed

Tables used (each folder): food, food_nutrient, nutrient, foundation_food / sr_legacy_food, food_category. Every join uses explicit
keys and `validate=`, and its row counts are logged. Nutrients are selected by `nutrient.nutrient_nbr` (255 Water, 204 Total lipid),
never by `nutrient.id`, because ids are not guaranteed to agree between releases. Amounts are g per 100 g.

Commodity -> FDC food matching uses ONLY the explicit COMMODITY_TO_FDC table below (exact description strings, written by hand and
meant to be reviewed). There is no fuzzy matching. A commodity mapped to None keeps its current value, with the reason recorded.
Foundation values are preferred over SR Legacy. pH is not in FDC and is left untouched.

Nothing here overwrites reference/commodities_master.csv: it writes a proposed table and a diff for approval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .commodities import load_commodities

NUTRIENT_NBR = {"water_g": 255, "fat_g": 204}
DISAGREE_PP = 3.0          # report |FDC - current| above this many percentage points

FOUNDATION_DIR = "FoodData_Central_foundation_food_csv_*"
SR_DIR = "FoodData_Central_sr_legacy_food_csv_*"


def _log(log: list[str], msg: str) -> None:
    log.append(msg)


def _join(left: pd.DataFrame, right: pd.DataFrame, log: list[str], what: str, **kw) -> pd.DataFrame:
    out = left.merge(right, **kw)
    _log(log, f"  join {what}: left {len(left):,} x right {len(right):,} -> {len(out):,} rows ({kw.get('how', 'inner')}, validate={kw.get('validate')})")
    return out


def load_folder(folder: Path, source: str, log: list[str]) -> pd.DataFrame:
    """One row per FDC food of this folder's own type, with water and fat (g/100 g) and their sample counts."""
    _log(log, f"{source}: {folder.name}")
    food = pd.read_csv(folder / "food.csv", usecols=["fdc_id", "data_type", "description", "food_category_id"], low_memory=False)
    _log(log, f"  food.csv: {len(food):,} rows; data_type counts {food['data_type'].value_counts().to_dict()}")
    own_type = "foundation_food" if source == "foundation" else "sr_legacy_food"
    food = food[food["data_type"] == own_type]
    _log(log, f"  kept data_type == {own_type}: {len(food):,}")
    listing = pd.read_csv(folder / f"{own_type}.csv", usecols=["fdc_id"])
    food = _join(food, listing.assign(in_listing=True), log, f"food -> {own_type}.csv", on="fdc_id", how="left", validate="one_to_one")
    missing = int(food["in_listing"].isna().sum())
    _log(log, f"  foods of type {own_type} not listed in {own_type}.csv: {missing} (kept; food.data_type is authoritative)")
    cats = pd.read_csv(folder / "food_category.csv", usecols=["id", "description"]).rename(columns={"id": "food_category_id", "description": "category"})
    food = _join(food, cats, log, "food -> food_category", on="food_category_id", how="left", validate="many_to_one")

    nut = pd.read_csv(folder / "nutrient.csv")
    sel = nut[nut["nutrient_nbr"].isin(NUTRIENT_NBR.values())][["id", "name", "unit_name", "nutrient_nbr"]]
    if len(sel) != len(NUTRIENT_NBR) or set(sel["unit_name"].str.upper()) != {"G"}:
        raise ValueError(f"{folder}: expected water and fat in G, found {sel.to_dict('records')}")
    _log(log, "  nutrients by nutrient_nbr: " + "; ".join(f"{r.nutrient_nbr:g} -> id {r.id} '{r.name}' ({r.unit_name})" for r in sel.itertuples()))
    fn = pd.read_csv(folder / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount", "data_points"], low_memory=False)
    _log(log, f"  food_nutrient.csv: {len(fn):,} rows")
    fn = _join(fn, sel.rename(columns={"id": "nutrient_id"}), log, "food_nutrient -> nutrient (water, fat only)", on="nutrient_id",
               how="inner", validate="many_to_one")
    fn = _join(fn, food[["fdc_id"]], log, f"-> {own_type} foods only", on="fdc_id", how="inner", validate="many_to_one")
    if fn.duplicated(["fdc_id", "nutrient_nbr"]).any():
        raise ValueError(f"{folder}: more than one amount per food and nutrient")
    key = {v: k for k, v in NUTRIENT_NBR.items()}
    fn["col"] = fn["nutrient_nbr"].map(key)
    amount = fn.pivot(index="fdc_id", columns="col", values="amount")
    points = fn.pivot(index="fdc_id", columns="col", values="data_points").add_suffix("_n")
    wide = pd.concat([amount, points], axis=1).reset_index()
    out = _join(food, wide, log, "foods -> water/fat", on="fdc_id", how="left", validate="one_to_one")
    out["source"] = "FDC Foundation" if source == "foundation" else "FDC SR Legacy"
    _log(log, f"  foods with water: {int(out['water_g'].notna().sum()):,}, with fat: {int(out['fat_g'].notna().sum()):,}")
    return out


# ------------------------------------------------------------------------------------------------- the alias table (REVIEW THIS)
# commodity -> (Foundation descriptions, SR Legacy descriptions, note). Exact FDC `description` strings. Several strings are
# averaged without weights (e.g. apple varieties). Foundation is used when its list is non-empty; SR Legacy otherwise (and for a
# nutrient Foundation lacks). Both lists empty = no confident match: the current value is kept and the note says why.
COMMODITY_TO_FDC: dict[str, tuple[list[str], list[str], str]] = {
    "Apple": (["Apples, red delicious, with skin, raw", "Apples, honeycrisp, with skin, raw", "Apples, granny smith, with skin, raw",
               "Apples, gala, with skin, raw", "Apples, fuji, with skin, raw"],
              ["Apples, raw, with skin (Includes foods for USDA's Food Distribution Program)"], "mean of 5 varieties, with skin"),
    "Asparagus": (["Asparagus, green, raw"], ["Asparagus, raw"], ""),
    "Avocado": (["Avocado, Hass, peeled, raw"], ["Avocados, raw, all commercial varieties"], "Hass only in Foundation"),
    "Banana": (["Bananas, ripe and slightly ripe, raw"], ["Bananas, raw"], "ripe, not overripe"),
    "Beet": (["Beets, raw"], ["Beets, raw"], ""),
    "Bell Pepper": (["Peppers, bell, green, raw", "Peppers, bell, yellow, raw", "Peppers, bell, red, raw", "Peppers, bell, orange, raw"],
                    ["Peppers, sweet, green, raw"], "mean of 4 colours"),
    "Blueberry": (["Blueberries, raw"], ["Blueberries, raw"], "cultivated, not wild"),
    "Brinjal": (["Eggplant, raw"], ["Eggplant, raw"], "brinjal = eggplant"),
    "Broccoli": (["Broccoli, raw"], ["Broccoli, raw"], ""),
    "Cabbage": (["Cabbage, green, raw"], ["Cabbage, raw"], "green (common) cabbage; red, napa and bok choy excluded"),
    "Carrot": (["Carrots, mature, raw"], ["Carrots, raw"], "mature, not baby"),
    "Cauliflower": (["Cauliflower, raw"], ["Cauliflower, raw"], ""),
    "Cucumber": (["Cucumber, with peel, raw"], ["Cucumber, with peel, raw"], ""),
    "Garlic": (["Garlic, raw"], ["Garlic, raw"], ""),
    "Grapes": (["Grapes, red, seedless, raw", "Grapes, green, seedless, raw"],
               ["Grapes, red or green (European type, such as Thompson seedless), raw"], "mean of red and green seedless"),
    "Green Chilli": ([], ["Peppers, hot chili, green, raw"],
                     "Foundation has only seeded jalapeno / serrano (specific cultivars, seeds removed); SR hot green chili used"),
    "Green Peas": ([], ["Peas, green, raw"], "Foundation has only canned peas"),
    "Guava": ([], ["Guavas, common, raw"], "not in Foundation; SR n = 2"),
    "Lettuce": (["Lettuce, iceberg, raw", "Lettuce, romaine, green, raw", "Lettuce, leaf, green, raw", "Lettuce, leaf, red, raw"],
                ["Lettuce, iceberg (includes crisphead types), raw"], "mean of iceberg, romaine, green leaf, red leaf (type not specified)"),
    "Mango": (["Mango, Tommy Atkins, peeled, raw", "Mango, Ataulfo, peeled, raw"], ["Mangos, raw"], "mean of 2 cultivars"),
    "Mushroom": (["Mushrooms, white button"], ["Mushrooms, white, raw"], "white button (most common); species not specified"),
    "Okra": ([], ["Okra, raw"], "not in Foundation"),
    "Onion": (["Onions, red, raw", "Onions, yellow, raw", "Onions, white, raw"], ["Onions, raw"], "mean of red, yellow, white"),
    "Orange": (["Oranges, raw, navels"], ["Oranges, raw, all commercial varieties"], "navel only in Foundation"),
    "Papaya": ([], ["Papayas, raw"], "not in Foundation"),
    "Peach": (["Peaches, yellow, raw"], ["Peaches, yellow, raw"], ""),
    "Pear": (["Pears, raw, bartlett", "Pear, Anjou, green, with skin, raw"], ["Pears, raw"], "mean of bartlett and anjou"),
    "Pineapple": (["Pineapple, raw"], ["Pineapple, raw, all varieties"], ""),
    "Pomegranate": ([], ["Pomegranates, raw"], "not in Foundation; SR n = 3"),
    "Potato": (["Potatoes, russet, without skin, raw", "Potatoes, red, without skin, raw", "Potatoes, gold, without skin, raw"],
               ["Potatoes, flesh and skin, raw"], "Foundation is without skin (mean of 3 types); whole tuber stored in practice"),
    "Spinach": (["Spinach, mature"], ["Spinach, raw"], "mature leaves (not baby)"),
    "Strawberry": (["Strawberries, raw"], ["Strawberries, raw"], ""),
    "Sweet corn": (["Corn, sweet, yellow and white kernels, \xa0fresh, raw"], ["Corn, sweet, yellow, raw"], "kernels"),
    "Sweet potato": (["Sweet potatoes, orange flesh, without skin, raw"],
                     ["Sweet potato, raw, unprepared (Includes foods for USDA's Food Distribution Program)"], "orange flesh"),
    "Tomato": ([], ["Tomatoes, red, ripe, raw, year round average"],
               "Foundation has only grape and roma tomatoes; SR red ripe year-round average is the generic tomato"),
    "Winter squash": (["Squash, winter, butternut, raw", "Squash, winter, acorn, raw"], ["Squash, winter, all varieties, raw"],
                      "mean of butternut and acorn"),
    # dry snacks: SR Legacy only (Foundation has almost no processed snacks)
    "Potato Chips": ([], ["Snacks, potato chips, plain, salted"], ""),
    "Tortilla Chips": ([], ["Snacks, tortilla chips, plain, white corn, salted"], ""),
    "Banana Chips": ([], ["Snacks, banana chips"], "SR n = 2"),
    "Extruded Puffs": ([], [], "NO MATCH: only cheese-flavoured puffs ('Snacks, corn-based, extruded, puffs or twists, cheese-flavor', "
                               "fat 36.5) or extruded chips; not the same product"),
    "Salted Crackers": ([], ["Crackers, standard snack-type, regular"],
                        "snack-type crackers (Monaco / Krackjack-like), chosen by Adi 2026-10-26; saltines would be fat 8.6"),
    "Cream Biscuits": ([], ["Cookies, vanilla sandwich with creme filling"], "cream biscuit = creme-filled sandwich cookie; vanilla"),
    "Namkeen Sev": ([], [], "NO MATCH: FDC has chickpea flour (besan) only, not fried sev"),
    "Roasted Peanuts": ([], ["Peanuts, all types, dry-roasted, with salt"], "dry-roasted (oil-roasted: fat 52.5 vs 49.7)"),
    "Roasted Almonds": (["Nuts, almonds, dry roasted, with salt added"], ["Nuts, almonds, dry roasted, with salt added"], ""),
    "Ready Popcorn": ([], ["Snacks, popcorn, oil-popped, white popcorn, salt added"],
                      "oil-popped salted; microwave popcorn would be fat 43.6, air-popped 4.5"),
    "Corn Flakes": ([], ["Cereals ready-to-eat, RALSTON Corn Flakes"], "the only corn flakes entry (brand: Ralston), SR n = 3"),
    "Instant Noodles": ([], ["Soup, ramen noodle, any flavor, dry"], "dry block including the seasoning packet"),
}


def _pick(table: pd.DataFrame, descriptions: list[str], source: str) -> pd.DataFrame:
    """Rows for exact descriptions. Identical descriptions (re-analyses published under new fdc_ids) -> the most recent fdc_id."""
    rows = []
    for d in descriptions:
        hit = table[table["description"] == d]
        if hit.empty:
            raise KeyError(f"alias table: {source} description not found: {d!r}")
        rows.append(hit.sort_values("fdc_id").iloc[-1])
    return pd.DataFrame(rows)


def composition(master: pd.DataFrame, foundation: pd.DataFrame, sr: pd.DataFrame) -> pd.DataFrame:
    missing = set(master["commodity"]) - set(COMMODITY_TO_FDC)
    if missing:
        raise KeyError(f"alias table has no entry for {sorted(missing)}")
    out = []
    for name in master["commodity"]:
        fo_desc, sr_desc, note = COMMODITY_TO_FDC[name]
        rec = dict(commodity=name, note=note)
        if not fo_desc and not sr_desc:
            out.append(rec | dict(match_status="no_match"))
            continue
        fo_rows = _pick(foundation, fo_desc, "Foundation") if fo_desc else None
        sr_rows = _pick(sr, sr_desc, "SR Legacy") if sr_desc else None
        rec["match_status"] = "matched"
        for col, nut in (("moisture_pct", "water_g"), ("fat_pct", "fat_g")):
            src_rows = fo_rows if fo_rows is not None and fo_rows[nut].notna().all() else sr_rows
            if src_rows is None or src_rows[nut].isna().any():
                rec[col], rec[f"{col}_source"] = np.nan, "missing"
                continue
            rec[col] = float(src_rows[nut].mean())
            rec[f"{col}_source"] = src_rows["source"].iloc[0]
            rec[f"{col}_n_samples"] = int(src_rows[f"{nut}_n"].fillna(0).sum())
            rec[f"{col}_fdc_id"] = ";".join(str(int(i)) for i in src_rows["fdc_id"])
            rec[f"{col}_description"] = " | ".join(src_rows["description"])
        rec["n_foods_averaged"] = len(fo_rows) if rec.get("moisture_pct_source") == "FDC Foundation" else len(sr_rows)
        out.append(rec)
    df = pd.DataFrame(out)
    # the columns asked for in T2, then the per-nutrient detail
    df["matched_fdc_description"] = df.get("moisture_pct_description")
    df["fdc_id"] = df.get("moisture_pct_fdc_id")
    df["n_samples"] = df.get("moisture_pct_n_samples")
    df["source"] = df.get("moisture_pct_source")
    first = ["commodity", "match_status", "matched_fdc_description", "fdc_id", "moisture_pct", "fat_pct", "n_samples", "source", "note"]
    return df[first + [c for c in df.columns if c not in first]]


def disagreements(master: pd.DataFrame, comp: pd.DataFrame, threshold: float = DISAGREE_PP) -> pd.DataFrame:
    m = master.set_index("commodity")
    rows = []
    for r in comp[comp["match_status"] == "matched"].itertuples():
        for col in ("moisture_pct", "fat_pct"):
            new, old = getattr(r, col), float(m.loc[r.commodity, col])
            if pd.notna(new) and abs(new - old) > threshold:
                rows.append(dict(commodity=r.commodity, column=col, current=old, fdc=round(new, 2), difference_pp=round(new - old, 2),
                                 current_trust=m.loc[r.commodity, "trust_composition"] if col == "moisture_pct" else "assumed (flat 0.3 for produce)"
                                 if m.loc[r.commodity, "category"] == "fresh_produce" else m.loc[r.commodity, "trust_composition"],
                                 fdc_source=getattr(r, f"{col}_source"), fdc_description=getattr(r, f"{col}_description")))
    return pd.DataFrame(rows)


def proposed_master(master: pd.DataFrame, comp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A proposed commodity table (moisture and fat from FDC where matched; pH and everything else unchanged) and its diff."""
    new = master.copy()
    c = comp.set_index("commodity")
    diff = []
    for i, r in new.iterrows():
        rec = c.loc[r["commodity"]]
        if rec["match_status"] != "matched":
            continue
        for col in ("moisture_pct", "fat_pct"):
            v = rec[col]
            if pd.notna(v) and round(float(v), 2) != float(r[col]):
                diff.append(dict(commodity=r["commodity"], column=col, current=r[col], proposed=round(float(v), 2),
                                 difference=round(float(v) - float(r[col]), 2), source=rec[f"{col}_source"], fdc_id=rec[f"{col}_fdc_id"]))
                new.at[i, col] = round(float(v), 2)
        srcs = {rec.get("moisture_pct_source"), rec.get("fat_pct_source")} - {None, "missing"}
        old = r["trust_composition"]
        new.at[i, "trust_composition"] = f"moisture+fat: {' / '.join(sorted(s.replace('FDC ', 'fdc_').lower().replace(' ', '_') for s in srcs))}; ph: {old}"
        diff.append(dict(commodity=r["commodity"], column="trust_composition", current=old, proposed=new.at[i, "trust_composition"],
                         difference=None, source="", fdc_id=""))
    return new, pd.DataFrame(diff)


def _find_dir(raw: Path, pattern: str) -> Path:
    hits = sorted(raw.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no folder matching {pattern} under {raw}")
    return hits[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="packr.fdc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    foundation = load_folder(_find_dir(a.raw, FOUNDATION_DIR), "foundation", log)
    sr = load_folder(_find_dir(a.raw, SR_DIR), "sr", log)
    master = load_commodities()
    comp = composition(master, foundation, sr)
    dis = disagreements(master, comp)
    new, diff = proposed_master(master, comp)
    comp.to_csv(a.out / "composition_from_fdc.csv", index=False)
    dis.to_csv(a.out / "composition_disagreements.csv", index=False)
    new.to_csv(a.out / "commodities_master_proposed.csv", index=False)
    diff.to_csv(a.out / "commodities_master_proposed_diff.csv", index=False)
    matched = comp[comp["match_status"] == "matched"]
    lines = log + ["", "COMPOSITION FROM FDC",
                   f"commodities: {len(comp)}; matched: {len(matched)} (Foundation for moisture: {(matched['source'] == 'FDC Foundation').sum()}, "
                   f"SR Legacy: {(matched['source'] == 'FDC SR Legacy').sum()}); no confident match: {(comp['match_status'] == 'no_match').sum()}",
                   "no match (value kept): " + "; ".join(f"{r.commodity} ({r.note})" for r in comp[comp['match_status'] == 'no_match'].itertuples()),
                   f"disagreements > {DISAGREE_PP:g} percentage points: {len(dis)}"]
    lines += [f"    {r.commodity:<16} {r.column:<13} current {r.current:>6.2f}  FDC {r.fdc:>6.2f}  ({r.difference_pp:+.2f})  [{r.fdc_source}]"
              for r in dis.itertuples()]
    lines += ["pH is not in FoodData Central: left unchanged.",
              "reference/commodities_master.csv NOT modified; proposed table: commodities_master_proposed.csv (+ _diff.csv)."]
    text = "\n".join(lines)
    (a.out / "composition_audit.txt").write_text(text + "\n", encoding="utf-8")
    enc = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, errors="backslashreplace").decode(enc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

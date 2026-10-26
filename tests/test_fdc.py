from pathlib import Path

import pandas as pd
import pytest

from packr import fdc
from packr.commodities import MASTER_CSV, load_commodities
from packr.paths import RAW_DIR


def write_folder(root: Path, kind: str, water_id: int, fat_id: int, foods: list[tuple]) -> Path:
    """A minimal FDC folder. `foods` = (fdc_id, description, water, fat, data_points). Nutrient ids differ between folders."""
    d = root / ("FoodData_Central_foundation_food_csv_x" if kind == "foundation" else "FoodData_Central_sr_legacy_food_csv_x")
    d.mkdir(parents=True)
    own = "foundation_food" if kind == "foundation" else "sr_legacy_food"
    rows = [dict(fdc_id=f[0], data_type=own, description=f[1], food_category_id=9, publication_date="2020") for f in foods]
    rows.append(dict(fdc_id=999999, data_type="sample_food", description=foods[0][1], food_category_id=9, publication_date="2020"))
    pd.DataFrame(rows).to_csv(d / "food.csv", index=False)
    pd.DataFrame({"fdc_id": [f[0] for f in foods]}).to_csv(d / f"{own}.csv", index=False)
    pd.DataFrame({"id": [9], "code": ["0900"], "description": ["Fruits and Fruit Juices"]}).to_csv(d / "food_category.csv", index=False)
    pd.DataFrame({"id": [water_id, fat_id, 1003], "name": ["Water", "Total lipid (fat)", "Protein"], "unit_name": ["G", "G", "G"],
                  "nutrient_nbr": [255, 204, 203], "rank": [1, 2, 3]}).to_csv(d / "nutrient.csv", index=False)
    fn = []
    for f in foods:
        fn.append(dict(id=len(fn), fdc_id=f[0], nutrient_id=water_id, amount=f[2], data_points=f[4]))
        if f[3] is not None:
            fn.append(dict(id=len(fn), fdc_id=f[0], nutrient_id=fat_id, amount=f[3], data_points=f[4]))
        fn.append(dict(id=len(fn), fdc_id=f[0], nutrient_id=1003, amount=1.0, data_points=f[4]))       # protein: must be ignored
    fn.append(dict(id=len(fn), fdc_id=999999, nutrient_id=water_id, amount=1.0, data_points=1))         # a sample row: must be ignored
    pd.DataFrame(fn).to_csv(d / "food_nutrient.csv", index=False)
    return d


@pytest.fixture
def folders(tmp_path):
    fo = write_folder(tmp_path, "foundation", 1051, 1004, [
        (10, "Apples, gala, with skin, raw", 84.0, 0.2, 8),
        (11, "Apples, fuji, with skin, raw", 86.0, 0.1, 8),
        (12, "Apples, fuji, with skin, raw", 85.0, 0.3, 8),           # re-analysis under a newer fdc_id: this one wins
        (20, "Cabbage, napa, leaf, destemmed, raw", 94.0, None, 8),   # no fat in Foundation
    ])
    sr = write_folder(tmp_path, "sr", 7777, 8888, [                   # different nutrient ids: selection must go by nutrient_nbr
        (30, "Apples, raw, with skin", 85.5, 0.17, 38),
        (31, "Cabbage, raw", 92.0, 0.1, 20),
    ])
    return fo, sr


def test_nutrients_are_selected_by_number_not_id_and_samples_are_excluded(folders):
    log = []
    sr = fdc.load_folder(folders[1], "sr", log)
    assert set(sr["fdc_id"]) == {30, 31}
    assert sr.set_index("fdc_id").loc[30, "water_g"] == 85.5 and sr.set_index("fdc_id").loc[30, "fat_g"] == 0.17
    assert any("255 -> id 7777" in line for line in log) and any("validate=many_to_one" in line for line in log)
    fo = fdc.load_folder(folders[0], "foundation", [])
    assert 999999 not in set(fo["fdc_id"])                              # sample_food rows dropped


def test_foundation_preferred_duplicates_deduplicated_and_missing_fat_falls_back(folders, monkeypatch):
    fo, sr = fdc.load_folder(folders[0], "foundation", []), fdc.load_folder(folders[1], "sr", [])
    monkeypatch.setattr(fdc, "COMMODITY_TO_FDC", {
        "Apple": (["Apples, gala, with skin, raw", "Apples, fuji, with skin, raw"], ["Apples, raw, with skin"], ""),
        "Cabbage": (["Cabbage, napa, leaf, destemmed, raw"], ["Cabbage, raw"], ""),
        "Mystery": ([], [], "NO MATCH"),
    })
    master = pd.DataFrame({"commodity": ["Apple", "Cabbage", "Mystery"], "category": ["fresh_produce"] * 3,
                           "moisture_pct": [84.0, 92.0, 50.0], "fat_pct": [0.3, 0.3, 1.0], "ph": [3.5, 6.0, 7.0],
                           "trust_composition": ["supplied_csv", "supplied_csv", "assumed"]})
    comp = fdc.composition(master, fo, sr).set_index("commodity")
    a = comp.loc["Apple"]
    assert a["source"] == "FDC Foundation" and a["moisture_pct"] == pytest.approx((84.0 + 85.0) / 2)   # fdc 12 used, not 11
    assert a["fdc_id"] == "10;12" and a["n_samples"] == 16
    c = comp.loc["Cabbage"]
    assert c["moisture_pct_source"] == "FDC Foundation" and c["fat_pct_source"] == "FDC SR Legacy" and c["fat_pct"] == 0.1
    assert comp.loc["Mystery", "match_status"] == "no_match"

    new, diff = fdc.proposed_master(master, comp.reset_index())
    assert new.set_index("commodity").loc["Mystery", "moisture_pct"] == 50.0                            # kept
    assert (new["ph"] == master["ph"]).all()                                                             # pH untouched
    assert "ph: supplied_csv" in new.set_index("commodity").loc["Apple", "trust_composition"]
    dis = fdc.disagreements(master, comp.reset_index())
    assert dis.empty or (dis["difference_pp"].abs() > fdc.DISAGREE_PP).all()


def test_missing_alias_description_is_an_error_not_a_fuzzy_match(folders, monkeypatch):
    fo, sr = fdc.load_folder(folders[0], "foundation", []), fdc.load_folder(folders[1], "sr", [])
    monkeypatch.setattr(fdc, "COMMODITY_TO_FDC", {"Apple": (["Apples, gala, raw"], [], "")})          # not the exact string
    master = pd.DataFrame({"commodity": ["Apple"], "category": ["fresh_produce"], "moisture_pct": [84.0], "fat_pct": [0.3],
                           "ph": [3.5], "trust_composition": ["supplied_csv"]})
    with pytest.raises(KeyError, match="not found"):
        fdc.composition(master, fo, sr)


def test_alias_table_covers_every_commodity():
    assert set(fdc.COMMODITY_TO_FDC) == set(load_commodities()["commodity"])


@pytest.mark.skipif(not any(RAW_DIR.glob(fdc.FOUNDATION_DIR)), reason="real FDC download not present")
def test_every_alias_exists_in_the_real_fdc_files_and_master_is_untouched():
    before = MASTER_CSV.read_bytes()
    fo = fdc.load_folder(fdc._find_dir(RAW_DIR, fdc.FOUNDATION_DIR), "foundation", [])
    sr = fdc.load_folder(fdc._find_dir(RAW_DIR, fdc.SR_DIR), "sr", [])
    comp = fdc.composition(load_commodities(), fo, sr)                  # raises if any alias string is missing
    assert (comp["match_status"] == "matched").sum() == 46                  # Salted Crackers mapped on 2026-10-26
    assert comp.set_index("commodity").loc["Avocado", "fat_pct"] > 15  # the flat 0.3 % assumed for produce was wrong here
    assert MASTER_CSV.read_bytes() == before


def test_master_table_carries_the_approved_fdc_composition():
    from packr.commodities import build_master
    m = build_master().set_index("commodity")
    assert m.loc["Avocado", "fat_pct"] == pytest.approx(20.31) and m.loc["Salted Crackers", "fat_pct"] == pytest.approx(26.43)
    assert m.loc["Avocado", "trust_composition"] == "moisture+fat: fdc_foundation; ph: assumed"
    assert m.loc["Namkeen Sev", "trust_composition"] == "assumed" and m.loc["Namkeen Sev", "fat_pct"] == 30.0   # no FDC match: kept
    assert (m["moisture_pct"] + m["fat_pct"] <= 100).all()

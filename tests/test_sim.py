import numpy as np
import pandas as pd
import pytest

from packr.commodities import build_master, load_commodities
from packr.sim.dataset import TEST_COMMODITIES, build
from packr.sim.physics import simulate

CM = load_commodities().set_index("commodity")


def row(name, **kw):
    r = CM.loc[name].to_dict()
    r.update(commodity=name, temp_c=25, rh_pct=70, harsh=0, target_days=180, mass_kg=0.1, area_m2_per_kg=1.0, otr23=1.5, wvtr38=1.0,
             light_barrier=0.95, seal=5, strength=4, perf_mm2_per_kg=0.0, flush_o2_pct=20.9)
    r.update(kw)
    return r


def sim(rows, n=256):
    return simulate(pd.DataFrame(rows), n_trials=n, seed=3)


def test_master_table_shape_and_trust():
    m = build_master()
    assert len(m) == 48 and m["commodity"].is_unique
    assert set(m["category"]) == {"fresh_produce", "dry_snack"}
    assert (m["trust_sim_params"] == "assumed").all()          # nothing in the simulator parameters is verified
    produce = m[m.category == "fresh_produce"]
    assert produce[["moisture_pct", "ph", "resp_ref_mgco2_kg_h", "base_life_days", "o2_min_pct"]].notna().all().all()


def test_snack_moisture_barrier_is_monotonic():
    r = sim([row("Potato Chips", wvtr38=w, rh_pct=80, flush_o2_pct=2) for w in (0.05, 1, 5, 20)])
    assert r["shelf_life_p50"].is_monotonic_decreasing
    assert r["success_rate"].iloc[0] > r["success_rate"].iloc[-1]


def test_nitrogen_flush_helps_a_foil_pack_oxidation():
    r = sim([row("Potato Chips", otr23=0.05, wvtr38=0.05, rh_pct=40, flush_o2_pct=f) for f in (20.9, 2.0)])
    assert r["shelf_life_p50"].iloc[1] > r["shelf_life_p50"].iloc[0]


def test_dry_cereal_is_not_oxidation_limited():
    r = sim([row("Corn Flakes", otr23=1000, wvtr38=0.05, rh_pct=40)])
    assert r["p_fail_oxidation"].iloc[0] == 0


def test_sealed_barrier_film_is_anaerobic_for_strawberry_but_open_pack_is_not():
    base = dict(temp_c=2, rh_pct=90, mass_kg=0.5, area_m2_per_kg=0.6, target_days=5, seal=3, strength=3, light_barrier=0.1)
    r = sim([row("Strawberry", otr23=10, wvtr38=3, **base), row("Strawberry", otr23=6500, wvtr38=18, perf_mm2_per_kg=30, **base)])
    assert r["p_fail_o2_low"].iloc[0] > 0.9 and r["x_o2_pct_p50"].iloc[0] < 1
    assert r["p_fail_o2_low"].iloc[1] < 0.05 and r["x_o2_pct_p50"].iloc[1] > 15


def test_hole_dominated_pack_has_o2_plus_co2_near_air():
    r = sim([row("Strawberry", otr23=100, wvtr38=3, perf_mm2_per_kg=5, temp_c=5, rh_pct=90, mass_kg=0.5, area_m2_per_kg=0.6, seal=3, strength=3)])
    assert 15 < r["x_o2_pct_p50"].iloc[0] + 0.8 * r["x_co2_pct_p50"].iloc[0] * 1.0 < 26


def test_hotter_means_shorter_life_and_success_rate_is_a_probability():
    r = sim([row("Tomato", temp_c=t, otr23=6500, wvtr38=18, perf_mm2_per_kg=2, rh_pct=90, mass_kg=0.5, area_m2_per_kg=0.6, target_days=10) for t in (8, 20, 32)])
    assert r["shelf_life_p50"].is_monotonic_decreasing
    assert r["success_rate"].between(0, 1).all()


def test_leak_prone_seals_hurt_a_high_barrier_pack():
    r = sim([row("Potato Chips", otr23=0.05, wvtr38=0.05, flush_o2_pct=2, rh_pct=80, seal=s, target_days=300) for s in (5, 1)], n=1024)
    assert r["success_rate"].iloc[0] > r["success_rate"].iloc[1]


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    build(out, seed=1, n_real_train=30, n_virtual_per_parent=2, n_rows_virtual=10, n_rows_test=30, n_trials_train=32, n_trials_test=32)
    return {k: pd.read_csv(out / f"{k}.csv.gz") for k in ("train", "test", "anchor_supplied", "anchor_cookie")}


def test_split_is_grouped_by_commodity_with_no_virtual_leak(quick):
    tr, te = quick["train"], quick["test"]
    assert set(te["commodity"]) == set(TEST_COMMODITIES)
    assert set(tr["parent_id"]).isdisjoint(te["parent_id"])
    assert not any(c.split("~")[0] in TEST_COMMODITIES for c in tr["commodity"])
    assert (te["source"] == "real").all() and (tr["source"] == "virtual").any()


def test_folds_are_by_parent_and_labels_are_valid(quick):
    tr = quick["train"]
    assert tr.groupby("parent_id")["fold"].nunique().max() == 1
    assert set(tr["fold"]) == set(range(5))
    for df in (tr, quick["test"]):
        assert df["success_rate"].between(0, 1).all() and np.isclose(df["success_rate"] + df["fail_rate"], 1).all()
        assert df[["otr23", "wvtr38", "target_days", "shelf_life_p50"]].notna().all().all() and (df["otr23"] > 0).all()
        assert df["row_id"].is_unique


def test_anchors_have_the_expected_rows(quick):
    assert len(quick["anchor_supplied"]) == 50 and len(quick["anchor_cookie"]) == 3

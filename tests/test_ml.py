import json

import numpy as np
import pandas as pd
import pytest

from packr.commodities import load_commodities
from packr.ml import features
from packr.ml.predict import load_model, predict
from packr.ml.train import calibration, estimated_profiles, fit_bundle, main, monotonic_wvtr_check
from packr.profile import nearest
from packr.sim import sufficiency
from packr.sim.dataset import TEST_COMMODITIES, build

TINY = dict(num_leaves=15, min_child_samples=20, n_estimators=120, learning_rate=0.1)


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    out = tmp_path_factory.mktemp("mlds")
    build(out, seed=2, n_real_train=40, n_virtual_per_parent=2, n_rows_virtual=20, n_rows_test=40, n_trials_train=32, n_trials_test=32)
    return pd.read_csv(out / "train.csv.gz"), pd.read_csv(out / "test.csv.gz")


@pytest.fixture(scope="module")
def bundle(quick):
    return fit_bundle(quick[0], TINY, TINY)


def test_feature_schema_is_stable():
    assert features.FEATURES == ["is_snack", "moisture_pct", "fat_pct", "ph", "resp_rate_mgco2_kg_h", "log_target_days", "temp_c", "rh_pct",
                                 "chilled", "harsh", "log_otr", "log_wvtr", "light_barrier", "seal", "strength", "thickness_um", "n_layers",
                                 "breathable", "mass_kg", "area_m2_per_kg", "log_perf", "flush_o2_pct"]
    assert features.LATENT == ["base_life_days", "q10_decay", "o2_min_pct", "co2_max_pct", "crit_weight_loss_pct", "ma_gain", "transp_g_kg_d",
                               "crit_aw", "iso_b", "ox_ref_days", "ea_ox_kj", "light_sens", "ideal_temp_c", "fragility"]
    assert sufficiency.FEATURES is features.FEATURES and sufficiency.LATENT is features.LATENT      # interface kept (spec section 10)


def test_featurise_with_profile_columns_and_nans(quick):
    x = features.featurise_with_profile(quick[0])
    assert list(x.columns) == features.FEATURES + features.LATENT
    snack = (quick[0]["category"] == "dry_snack").to_numpy()
    assert x.loc[snack, "o2_min_pct"].isna().all() and x.loc[~snack, "crit_aw"].isna().all()     # NaN, not a sentinel


def test_bundle_predictions_are_valid(bundle, quick):
    test = quick[1]
    assert bundle["columns"] == features.FEATURES + features.LATENT
    assert not set(bundle["trained_on"]) & set(TEST_COMMODITIES)
    p = predict(bundle, test)
    assert p["success_rate"].between(0, 1).all() and np.allclose(p["success_rate"] + p["fail_rate"], 1)
    assert (p["shelf_life_p10"] <= p["shelf_life_p50"]).all() and (p["shelf_life_p50"] <= p["shelf_life_p90"]).all()


def test_higher_wvtr_does_not_raise_snack_success_on_average(bundle, quick):
    assert monotonic_wvtr_check(bundle, quick[1])["passed"]


def test_longer_target_never_raises_predicted_success(bundle, quick):
    test = quick[1]
    longer = test.assign(target_days=test["target_days"] * 2)
    assert (predict(bundle, longer)["success_rate"].to_numpy() <= predict(bundle, test)["success_rate"].to_numpy() + 1e-9).all()


def test_estimated_profiles_use_only_train_commodities(quick):
    train, test = quick
    train_names = sorted(train.loc[train["source"] == "real", "commodity"].unique())
    master = load_commodities()
    est = estimated_profiles(test, master, train_names)
    pool = master[master["commodity"].isin(train_names)]
    for name, grp in est.groupby("commodity"):
        r = master.set_index("commodity").loc[name]
        nn = nearest(pool, r["category"], r["moisture_pct"], r["fat_pct"],
                     resp_ref=r["resp_ref_mgco2_kg_h"] if r["category"] == "fresh_produce" else None)
        assert name not in set(nn["commodity"]) and set(nn["commodity"]) <= set(train_names)
        assert np.allclose(grp["base_life_days"], nn["base_life_days"].mean())
    pd.testing.assert_frame_equal(features.featurise(est), features.featurise(test))       # plan inputs untouched


def test_calibration_bins_cover_every_row():
    y = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    cal = calibration(y, y)
    assert len(cal) == 10 and sum(b["n"] for b in cal) == 5
    assert all(b["abs_err"] < 1e-12 for b in cal if b["n"])


def test_no_test_build_puts_every_commodity_in_train(tmp_path):
    s = build(tmp_path, seed=3, n_real_train=5, n_virtual_per_parent=1, n_rows_virtual=3, n_rows_test=5, n_trials_train=8,
              n_trials_test=8, test_commodities=[])
    tr, te = pd.read_csv(tmp_path / "train.csv.gz"), pd.read_csv(tmp_path / "test.csv.gz")
    assert s["n_test"] == 0 and te.empty and list(te.columns) == list(tr.columns)
    assert tr.loc[tr["source"] == "real", "commodity"].nunique() == 48
    # the final refit reads that empty test file without breaking the dtypes
    (tmp_path / "params.json").write_text(json.dumps({"success_params": TINY, "life_params": TINY}))
    main(["--final", "--data", str(tmp_path), "--params", str(tmp_path / "params.json"), "--models", str(tmp_path / "m")])
    assert len(load_model(tmp_path / "m" / "packr_gbt.joblib")["trained_on"]) == 48

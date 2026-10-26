"""Empirical GAB fit (B1e). Each test states its hand-computed expectation."""
import numpy as np
import pandas as pd
import pytest

from packr.paths import REFERENCE_DIR
from packr.sim import calibrate, sorption_fit
from packr.sim.physics import gab_moisture


def test_effective_variance_residual_hand_value():
    # m0 0.05, C 10, K 0.9 at aw 0.3: G = 0.053916; G' = m0 C K (d - aw d') / d^2 with d = 0.73 x 3.43 = 2.5039,
    # d' = -0.9 x 3.43 + 0.73 x 9 x 0.9 = 2.826 -> G' = 0.45 x (2.5039 - 0.8478) / 6.26952 = 0.118868
    # point M 0.060, sd_M 0.002, sd_aw 0.01: r = 0.006084 / sqrt(0.002^2 + (0.118868 x 0.01)^2) = 0.006084 / 0.0023266 = 2.6150
    r = sorption_fit._residuals(np.log([0.05, 10.0, 0.9]), np.array([0.060]), np.array([0.3]), np.array([0.002]), np.array([0.01]))
    assert r[0] == pytest.approx(2.6150, rel=1e-3)


def test_fit_recovers_known_constants_from_exact_well_spread_points():
    aw = np.linspace(0.1, 0.85, 12)
    pts = pd.DataFrame({"label": [f"p{i}" for i in range(12)], "moisture_db": gab_moisture(aw, 0.06, 12.0, 0.85), "aw": aw,
                        "sd_m": 0.001, "sd_aw": 0.001})
    f = sorption_fit.fit_gab(pts)
    assert (f["m0"], f["C"], f["K"]) == pytest.approx((0.06, 12.0, 0.85), rel=1e-3)
    assert f["chi2"] == pytest.approx(0.0, abs=1e-6)


def test_banana_points_are_the_initial_pair_plus_15_films_with_median_sd_for_the_initial():
    pts = sorption_fit.points(calibrate.load_pairings())
    assert len(pts) == 16 and pts.loc[0, "label"] == "initial"
    assert (pts.loc[0, "moisture_db"], pts.loc[0, "aw"]) == pytest.approx((0.026, 0.16))
    films = pts.iloc[1:]
    assert pts.loc[0, "sd_m"] == pytest.approx(films["sd_m"].median()) and pts.loc[0, "sd_aw"] == pytest.approx(films["sd_aw"].median())
    assert (pts["aw"].min(), pts["aw"].max()) == (0.16, 0.70)
    assert len(sorption_fit.points(calibrate.load_pairings(), exclude=("BC-M",))) == 15


def test_stored_banana_curve_is_flagged_unidentifiable_and_range_limited():
    t = pd.read_csv(REFERENCE_DIR / "sorption_params.csv", keep_default_na=False).set_index("commodity")
    r = t.loc["Banana Chips"]
    assert r["trust"] == sorption_fit.TRUST_UNIDENTIFIABLE and (float(r["valid_aw_min"]), float(r["valid_aw_max"])) == (0.16, 0.70)
    assert r["crit_aw"] == "" and r["doi"] == "10.1007/s13197-012-0839-0" and "only the curve" in r["notes"].lower()


def test_identifiability_rule():
    good = pd.DataFrame({"m0": np.linspace(0.05, 0.06, 50), "C": np.linspace(10, 12, 50)[::-1] * 1.0, "K": np.full(50, 0.8)})
    good["K"] = 0.8 + 0.01 * np.sin(np.arange(50))                     # not on a bound, narrow, not collinear with m0
    assert sorption_fit.identifiability(good)["identifiable"] is False   # m0 and C exactly anti-correlated (|r| = 1 > 0.9)
    rng = np.random.default_rng(0)
    indep = pd.DataFrame({"m0": 0.05 * np.exp(rng.normal(0, 0.05, 200)), "C": 10 * np.exp(rng.normal(0, 0.05, 200)),
                          "K": 0.8 * np.exp(rng.normal(0, 0.02, 200))})
    assert sorption_fit.identifiability(indep)["identifiable"] is True
    wide = indep.assign(K=np.where(np.arange(200) < 20, 1.0, indep["K"]))   # 10 % of resamples on the K = 1 bound
    assert any("K on a search bound in 10%" in s for s in sorption_fit.identifiability(wide)["reasons"])


def test_leave_one_film_out_refits_without_the_held_out_film():
    p = calibrate.load_pairings()
    keep = ["BC-A", "BC-D", "BC-M", "BC-N"]
    sub = p[~p["pairing_id"].str.startswith("BC-") | p["pairing_id"].isin(keep)]
    loo = calibrate.loo_films(sub)
    assert loo["pairing_id"].tolist() == keep
    # the fold's valid aw max is the max of the OTHER points: holding out BC-D (aw 0.70) leaves BC-A/BC-M 0.59 as the max
    assert loo.set_index("pairing_id")["fold_aw_max"].to_dict() == {"BC-A": 0.70, "BC-D": 0.59, "BC-M": 0.70, "BC-N": 0.70}
    for r in loo.itertuples():
        f = sorption_fit.fit_gab(sorption_fit.points(sub, exclude=(r.pairing_id,)))
        assert (r.fold_m0, r.fold_C, r.fold_K) == pytest.approx((f["m0"], f["C"], f["K"]))

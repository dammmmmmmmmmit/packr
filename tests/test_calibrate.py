import json

import pandas as pd
import pytest

from packr.sim import calibrate
from packr.sim.physics import Constants


def pairing(**kw):
    p = dict(pairing_id="T1", commodity_in_table="Cream Biscuits", food_description="cookie", film="PP film", otr_cm3_m2_d=1300,
             wvtr_g_m2_d=5.0, thickness_um="", atmosphere="air", temp_c="22", rh_pct="80", pack_mass_g="100", pack_area_cm2="1500",
             endpoint="texture_sensory", observed_value="compromised", observed_weeks=26, outcome_pass=False, source="test", doi="",
             verified_by_human=True, quality_notes="", trust="test")
    p.update(kw)
    return pd.Series(p)


def test_the_real_pairings_file_has_the_required_columns_and_is_unverified():
    df = calibrate.load_pairings()
    need = {"pairing_id", "commodity_in_table", "food_description", "film", "otr_cm3_m2_d", "wvtr_g_m2_d", "thickness_um", "atmosphere",
            "temp_c", "rh_pct", "pack_mass_g", "pack_area_cm2", "endpoint", "observed_value", "observed_weeks", "outcome_pass", "source",
            "doi", "verified_by_human", "quality_notes", "trust"}
    assert need <= set(df.columns) and len(df) == 8 + 15                       # 8 first pairings + 15 Manikantan et al. 2012 films
    assert not df["verified_by_human"].any()                                   # all tentative until a human checks them


def test_banana_chip_rows_carry_the_paper_facts_and_only_text_confirmed_observations():
    bc = calibrate.load_pairings().set_index("pairing_id").filter(like="BC-", axis=0)
    assert len(bc) == 15 and set(bc["doi"]) == {"10.1007/s13197-012-0839-0"}
    assert set(bc["initial_moisture_db_pct"]) == {"2.6"} and set(bc["pack_mass_g"]) == {"50"} and set(bc["pack_area_cm2"]) == {"300"}
    assert (bc["film_test_temp_c"] == "").all()                                 # WVTR test conditions not stated
    # all 15 films' day-120 values (automated reader, cross-checked against the text, not against the PDF)
    assert (bc["obs_moisture_db_pct"] != "").all() and (bc["obs_aw"] != "").all() and (bc["obs_moisture_sd"] != "").all()
    assert bc.loc["BC-N", ["obs_moisture_db_pct", "obs_aw"]].tolist() == ["5.3", "0.48"] and bc.loc["BC-M", "obs_moisture_db_pct"] == "9.3"
    assert float(bc.loc["BC-B", "obs_aw"]) > 0.60 and float(bc.loc["BC-D", "obs_aw"]) == 0.70   # agree with the text-confirmed values


def test_wvtr_from_an_unknown_test_temperature_is_restated_at_38c():
    # 38 C: unchanged. 23 C / 90 %: p_sat(38)/p_sat(23) = 6.62454/2.80935 = 2.35805; Arrhenius(25 kJ, 23 vs 38) = 0.612944;
    # factor = 2.35805 / 0.612944 = 3.8471
    assert calibrate.wvtr38_equivalent(2.0, 38.0) == pytest.approx(2.0, rel=1e-12)
    assert calibrate.wvtr38_equivalent(1.0, 23.0) == pytest.approx(3.8471, rel=1e-4)


def test_continuous_rows_sweep_ambient_and_film_test_temperature_from_the_measured_start():
    p = calibrate.load_pairings().set_index("pairing_id").loc["BC-N"].copy()
    p["pairing_id"] = "BC-N"
    rows = calibrate.continuous_rows(p, calibrate.load_commodities())
    assert len(rows) == 3 * 3 * 3                                              # T 20.2..32.5 x RH 50.8..71.3 x WVTR test T 23/30.5/38
    assert set(rows["temp_c"]) == {20.2, 26.35, 32.5} and set(rows["film_test_t"]) == set(calibrate.SWEEP_FILM_TEST_T_C)
    assert rows["moisture_pct"].iloc[0] == pytest.approx(100 * 0.026 / 1.026)  # 2.6 % d.b. -> 2.5341 % w.b.
    assert rows["area_m2_per_kg"].iloc[0] == pytest.approx(0.03 / 0.05)        # 300 cm2 over 50 g
    assert not calibrate.score_all(pd.DataFrame([p])).shape[0]                 # no pass/fail end point: not in the binary score


def test_observation_against_interval():
    assert calibrate._compare("9.3", 3.8, 6.4, 1) == "under-predicted by 2.9 or more"
    assert calibrate._compare("2.0", 3.8, 6.4, 1) == "over-predicted by 1.8 or more"
    assert calibrate._compare("5.3", 3.2, 6.4, 1) == "within interval"
    assert calibrate._compare(">0.60", 0.65, 0.70, 2) == "consistent"
    assert calibrate._compare(">0.60", 0.40, 0.60, 2) == "inconsistent"
    assert calibrate._compare(">0.60", 0.40, 0.71, 2) == "undetermined"
    assert calibrate._compare("", 0.4, 0.7, 2) == ""


def test_isotherm_check_points_use_v1_linear_aw():
    iso = calibrate.isotherm_checks()
    # Banana Chips iso_b 0.09: aw = 0.026 / 0.09 = 0.2889 (measured 0.16); 0.053 / 0.09 = 0.5889 (measured 0.48)
    assert iso["aw_v1"].round(4).tolist() == [0.2889, 0.5889] and iso["aw_measured"].tolist() == [0.16, 0.48]
    assert iso["aw_b1"].isna().all()                                           # sorption_params.csv has no sourced row


def test_unknown_inputs_are_swept_not_fixed():
    p = pairing(rh_pct="", pack_mass_g="", temp_c="15-20", pack_area_cm2="550", commodity_in_table="Potato Chips", endpoint="moisture_db_pct 3.0")
    rows = calibrate.scenario_rows(p, calibrate.load_commodities())
    assert len(rows) == 3 * 3 * 3                                              # T 15/17.5/20 x RH 30/55/80 x fill 20/40/60 g
    assert set(rows["rh_pct"]) == set(calibrate.SWEEP_RH_PCT) and set(rows["temp_c"]) == {15.0, 17.5, 20.0}
    # area per kg follows the fixed bag area: 550 cm2 = 0.055 m2 over the swept mass
    assert sorted(rows["area_m2_per_kg"].round(4).unique()) == sorted({round(0.055 / (m / 1000), 4) for m in calibrate.SWEEP_FILL_G_PER_600CM3})


def test_verdict_follows_the_observation_and_needs_the_whole_interval():
    master = calibrate.load_commodities()
    # cookie at RH 80 % > crit_aw 0.55: texture is lost within weeks -> "reached" observation is consistent
    wet = calibrate.score_pairing(pairing(rh_pct="80"), Constants(), master)
    assert wet["n_sweep"] == 1 and wet["verdict"] == "consistent" and wet["p_reached_lo"] > 0.5
    # RH 40 % < crit_aw: never reached -> the same observation is inconsistent; a "not reached" observation is consistent
    dry = calibrate.score_pairing(pairing(rh_pct="40"), Constants(), master)
    assert dry["verdict"] == "inconsistent"
    assert calibrate.score_pairing(pairing(rh_pct="40", outcome_pass=True), Constants(), master)["verdict"] == "consistent"
    # sweeping RH across the critical aw leaves it undetermined
    assert calibrate.score_pairing(pairing(rh_pct=""), Constants(), master)["verdict"] == "undetermined"


def test_censored_observation_only_penalises_a_shorter_simulated_life():
    s = calibrate.score_pairing(pairing(rh_pct="40", outcome_pass=True), Constants(), calibrate.load_commodities())
    assert s["logerr_hi"] <= 0.0 and s["logerr_lo"] == 0.0                   # simulated life (never) >= observed 26 weeks


def test_fitting_refuses_without_verified_pairings_or_sourced_priors():
    priors = pd.DataFrame({"parameter": ["crit_aw"], "low": [0.3], "high": [0.6], "doi": ["10.x/test"]})
    two = pd.DataFrame([pairing(pairing_id="a"), pairing(pairing_id="b")])
    assert calibrate.fit_lopo(two, priors, ["crit_aw"])["status"] == "not run"
    three = pd.DataFrame([pairing(pairing_id=i) for i in "abc"])
    assert "no sourced prior" in calibrate.fit_lopo(three, priors.iloc[:0], ["crit_aw"])["reason"]


def test_unsourced_prior_rows_are_ignored(tmp_path):
    f = tmp_path / "priors.csv"
    pd.DataFrame({"parameter": ["crit_aw", "iso_b"], "category": ["dry_snack"] * 2, "commodity": ["", ""], "low": [0.3, 0.05],
                  "high": [0.6, 0.15], "basis": ["", ""], "doi": ["10.1/x", ""], "notes": ["", ""], "trust": ["", ""]}).to_csv(f, index=False)
    assert calibrate.load_priors(f)["parameter"].tolist() == ["crit_aw"]


def test_leave_one_out_recovers_a_known_parameter():
    # synthetic truth: crit_aw = 0.40. Each pairing's observed time is the simulated median time under the truth, so the
    # midpoint log error is 0 exactly at 0.40, which is on the 7-point grid 0.30, 0.35, ..., 0.60.
    master = calibrate.load_commodities()
    pairs = []
    for i, (rh, w) in enumerate(((60, 5.0), (70, 2.0), (80, 8.0), (65, 4.0))):
        p = pairing(pairing_id=f"S{i}", rh_pct=str(rh), wvtr_g_m2_d=w)
        weeks = calibrate.score_pairing(p, Constants(), master, overrides={"crit_aw": 0.40})["life_weeks_lo"]
        p["observed_weeks"] = weeks
        pairs.append(p)
    priors = pd.DataFrame({"parameter": ["crit_aw"], "low": [0.30], "high": [0.60], "doi": ["10.x/test"]})
    res = calibrate.fit_lopo(pd.DataFrame(pairs), priors, ["crit_aw"])
    assert res["status"] == "run" and len(res["folds"]) == 4
    assert all(f["fitted"]["crit_aw"] == pytest.approx(0.40) for f in res["folds"])
    assert all(abs(f["held_out_logerr"]) < 0.05 for f in res["folds"])


def test_top_sensitive_skips_sourced_inputs(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"dry_snack": [dict(factor="moisture_pct", rank_success=1, sourced=True),
                                           dict(factor="iso_b", rank_success=2, sourced=False),
                                           dict(factor="Constants.qcrit_ml_per_g_fat", rank_success=3, sourced=False),
                                           dict(factor="crit_aw", rank_success=4, sourced=False),
                                           dict(factor="ox_ref_days", rank_success=5, sourced=False)]}))
    assert calibrate.top_sensitive("dry_snack", path=f) == ["iso_b", "qcrit_ml_per_g_fat", "crit_aw"]

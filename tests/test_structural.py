"""Structural options of the simulator (spec T3 part B). Each test states its hand-computed expectation."""
import math

import numpy as np
import pandas as pd
import pytest

from packr.commodities import load_commodities
from packr.materials import layer_rh, otr_rh_multiplier
from packr.sim import physics
from packr.sim.physics import (CONSTANTS, Constants, gab_aw, gab_moisture, o2_transient, simulate, time_to_moisture_gab)

CM = load_commodities().set_index("commodity")
QUIET = dict(tol_sigma_ln=1e-9, sigma_t_normal=1e-9, rh_sigma=1e-9)        # remove Monte Carlo spread for exact checks


def row(name, **kw):
    r = CM.loc[name].to_dict()
    r.update(commodity=name, temp_c=25, rh_pct=70, harsh=0, target_days=180, mass_kg=0.1, area_m2_per_kg=1.0, otr23=1.5, wvtr38=1.0,
             light_barrier=0.95, seal=5, strength=4, perf_mm2_per_kg=0.0, flush_o2_pct=20.9)
    r.update(kw)
    return r


def sim(rows, n=512, c=CONSTANTS, seed=3):
    return simulate(pd.DataFrame(rows), n_trials=n, seed=seed, c=c, extra=True)


# ------------------------------------------------------------------------------------------------ v1 is the default

def test_default_constants_are_v1_and_structural_fields_are_tagged():
    assert CONSTANTS.is_v1 and not Constants(isotherm="gab").is_v1
    from dataclasses import fields
    tagged = {f.name for f in fields(Constants) if f.metadata.get("structural")}
    assert {"isotherm", "oxidation", "rh_dependent_otr", "temp_history", "produce_gas", "excursion_rate_harsh"} <= tagged
    assert "iso_b" not in tagged and "qcrit_ml_per_g_fat" not in tagged


def test_options_without_sourced_tables_fall_back_to_v1_and_say_so():
    rows = [row("Potato Chips"), row("Strawberry", temp_c=4, rh_pct=90, target_days=5, mass_kg=0.5, area_m2_per_kg=0.6, otr23=6500)]
    base = sim(rows)
    for opt in (dict(isotherm="gab"), dict(oxidation="pv"), dict(rh_dependent_otr=True)):
        r = sim(rows, c=Constants(**opt))
        pd.testing.assert_frame_equal(r[base.columns], base)
        assert "sim_model" in r and "sim_model" not in base
    note = sim(rows, c=Constants(isotherm="gab", oxidation="pv"))["sim_model"].iloc[0]
    assert "fallback: no sourced GAB row" in note and "fallback: no sourced oil stability row" in note


# ------------------------------------------------------------------------------------------------ B1 GAB

def test_gab_moisture_hand_value_and_inverse():
    # m0 0.05, C 10, K 0.9, aw 0.3: M = 0.05*10*0.9*0.3 / ((1-0.27)(1-0.27+2.7)) = 0.135 / (0.73*3.43) = 0.135 / 2.5039 = 0.053916
    assert gab_moisture(0.3, 0.05, 10.0, 0.9) == pytest.approx(0.053916, rel=1e-4)
    assert gab_aw(0.053916, 0.05, 10.0, 0.9) == pytest.approx(0.3, abs=1e-4)
    aw = np.linspace(0.02, 0.95, 20)
    assert np.allclose(gab_aw(gab_moisture(aw, 0.06, 15.0, 0.85), 0.06, 15.0, 0.85), aw, atol=1e-9)


def test_gab_puts_a_4_3_percent_moisture_snack_well_below_aw_0_5():
    # 4.3 % wet basis = 4.3/95.7 = 0.04493 g/g dry. GAB (0.05, 10, 0.9): M(0.22) = 0.099/(0.802*2.782) = 0.04437,
    # M(0.225) = 0.10125/(0.7975*2.8225) = 0.04498  ->  aw0 ~ 0.2246. The v1 linear isotherm (iso_b 0.09) gives 0.04493/0.09 = 0.499.
    assert gab_aw(4.3 / 95.7, 0.05, 10.0, 0.9) == pytest.approx(0.2246, abs=0.001)
    chips = row("Banana Chips", moisture_pct=4.3, rh_pct=60, wvtr38=0.5)
    v1 = sim([chips])
    gab = sim([chips | dict(gab_m0=0.05, gab_C=10.0, gab_K=0.9)], c=Constants(isotherm="gab"))
    assert v1["aw0_p50"].iloc[0] == pytest.approx(0.499, abs=0.002)
    assert gab["aw0_p50"].iloc[0] == pytest.approx(0.2246, abs=0.001)
    assert gab["life_moisture_p50"].iloc[0] > 10 * max(v1["life_moisture_p50"].iloc[0], 1.0)
    assert gab["sim_model"].iloc[0].startswith("iso=gab")


def test_gab_time_matches_a_fine_direct_integration_including_the_stiff_case():
    # dt = dM / (k (RH - aw(M))). Direct: 200,000-step midpoint rule in M. RH just above the critical aw is the stiff case.
    m0, c, k, kef = 0.05, 10.0, 0.9, 0.002
    for rh, awc in ((0.80, 0.50), (0.505, 0.50)):
        aw0 = 0.2
        ms = np.linspace(gab_moisture(aw0, m0, c, k), gab_moisture(awc, m0, c, k), 200_001)
        mid = 0.5 * (ms[1:] + ms[:-1])
        direct = np.sum(np.diff(ms) / (kef * (rh - gab_aw(mid, m0, c, k))))
        quad = time_to_moisture_gab(np.array([aw0]), np.array([awc]), np.array([rh]), np.array([kef]), m0, c, k)[0]
        assert quad == pytest.approx(direct, rel=2e-3)
    assert np.isinf(time_to_moisture_gab(np.array([0.2]), np.array([0.5]), np.array([0.45]), np.array([kef]), m0, c, k))[0]


def test_critical_moisture_override_matches_the_linear_closed_form():
    # v1 linear isotherm, quiet trials, no leak area: t = ln((Me - M0)/(Me - Mc)) / K, Me = b RH, K = C_wv/(Ws b),
    # C_wv = WVTR38 * arrhenius(25 vs 38 C) / dp38 * area * psat(25)
    r = row("Potato Chips", rh_pct=70, wvtr38=1.0, crit_moisture_db=0.03, seal=5, fragility=0.0)
    out = sim([r], c=Constants(**QUIET), n=200)
    b, m0 = r["iso_b"], r["moisture_pct"] / (100 - r["moisture_pct"])
    cwv = 1.0 * math.exp(-25.0 / 8.314e-3 * (1 / 298.15 - 1 / 311.15)) / physics.WVTR_DP_KPA * 0.1 * physics.psat_kpa(25.0)
    ws = 100.0 * (1 - r["moisture_pct"] / 100)
    expect = math.log((b * 0.70 - m0) / (b * 0.70 - 0.03)) / (cwv / (ws * b))
    assert out["life_moisture_p50"].iloc[0] == pytest.approx(expect, rel=1e-3)


# ------------------------------------------------------------------------------------------------ B2 PV oxidation

PV_OIL = dict(oil_pv_rate_25c=0.1, oil_ea_kj=1e-6, oil_km_o2=0.01)     # 0.1 mEq/kg/day in air at 25 C, flat in T, Km 1 % O2
NO_MOISTURE = dict(wvtr38=1e-9, rh_pct=20, light_sens=0.0, fragility=0.0)


def test_pv_end_point_in_constant_air_is_endpoint_over_rate():
    # Huge headspace keeps pO2 = 0.209, so PV grows at 0.1 mEq/kg/day: PV 10 at 10 / 0.1 = 100 days (median over lot variability).
    out = sim([row("Potato Chips", **NO_MOISTURE, **PV_OIL)], c=Constants(oxidation="pv", headspace_cm3_per_g=1e7))
    assert out["life_oxidation_p50"].iloc[0] == pytest.approx(100.0, rel=0.03)
    assert out["sim_model"].iloc[0].split("|")[1] == "ox=pv"


def test_pv_rate_follows_michaelis_menten_in_headspace_o2():
    # pO2 held at 1 % = Km: rate factor f(0.01)/f(0.209) = 0.5 / (0.209/0.219) = 0.5239  ->  100 / 0.5239 = 190.9 days
    out = sim([row("Potato Chips", flush_o2_pct=1.0, otr23=1e-9, **NO_MOISTURE, **PV_OIL)], c=Constants(oxidation="pv", headspace_cm3_per_g=1e7))
    assert out["life_oxidation_p50"].iloc[0] == pytest.approx(190.9, rel=0.03)


def test_pv_end_point_is_unreachable_when_the_sealed_pack_holds_too_little_oxygen():
    # 100 g, 34 % fat = 0.034 kg oil. PV 10 needs 10 * 0.034 = 0.34 mEq = 0.17 mmol O2 = 0.17 * 22.414 = 3.81 cm3 O2.
    # Headspace 2.5 cm3/g * 100 g = 250 cm3: at 1 % O2 = 2.5 cm3 (never enough), at 2 % = 5.0 cm3 (enough). No ingress, seal 5.
    common = dict(otr23=1e-9, fat_pct=34.0, **NO_MOISTURE, **PV_OIL)
    low = sim([row("Potato Chips", flush_o2_pct=1.0, **common)], c=Constants(oxidation="pv"))
    high = sim([row("Potato Chips", flush_o2_pct=2.0, **common)], c=Constants(oxidation="pv"))
    assert low["life_oxidation_p50"].iloc[0] == physics.LIFE_CAP_DAYS
    assert low["p_oxidation_by_target"].iloc[0] <= 0.05                     # only leaking trials (seal 5: 1 %) can get there
    assert high["life_oxidation_p50"].iloc[0] < physics.LIFE_CAP_DAYS


# ------------------------------------------------------------------------------------------------ B3 RH-dependent OTR

MATS = pd.DataFrame({"name": ["A", "B"], "OTR": [100.0, 1.0], "WVTR": [10.0, 10.0], "barrier_type": ["fixed", "fixed"]}).set_index("name")


def test_layer_rh_interpolates_in_water_vapour_resistance():
    # equal resistances 0.1 + 0.1: mid-points at 1/4 and 3/4 of the way from outside (80 %) to inside (20 %): 65 % and 35 %
    assert layer_rh(["A", "B"], rh_in=20, rh_out=80, materials=MATS) == pytest.approx([65.0, 35.0])


def test_otr_multiplier_of_a_laminate_hand_value():
    # B's curve: x1 at 0 % RH, x3 at 100 % RH -> x1.7 at 35 %. Nominal OTR 1/(1/100 + 1/1) = 0.990099;
    # at RH 1/(1/100 + 1/1.7) = 1/0.598235 = 1.671582; ratio 1.671582 / 0.990099 = 1.688298
    table = pd.DataFrame({"material": ["B", "B"], "rh_pct": [0, 100], "otr_multiplier": [1.0, 3.0]})
    assert otr_rh_multiplier(["A", "B"], 20, 80, MATS, table) == pytest.approx(1.688298, rel=1e-6)
    assert otr_rh_multiplier(["A", "B"], 20, 80, MATS, table.iloc[:0]) == 1.0


def test_otr_rh_factor_reaches_the_simulator():
    # produce film conductance doubles -> in-pack O2 rises
    base = dict(temp_c=4, rh_pct=90, target_days=5, mass_kg=0.5, area_m2_per_kg=0.6, otr23=3000, wvtr38=10, seal=3, strength=3)
    a = sim([row("Strawberry", **base)], c=Constants(rh_dependent_otr=True))
    b = sim([row("Strawberry", otr_rh_factor=2.0, **base)], c=Constants(rh_dependent_otr=True))
    assert b["x_o2_pct_p50"].iloc[0] > a["x_o2_pct_p50"].iloc[0] + 1.0


# ------------------------------------------------------------------------------------------------ B4 temperature excursions

def test_time_average_of_an_arrhenius_rate_hand_value():
    # one excursion of +10 K for 10 % of the time at 25 C, Ea 80 kJ/mol:
    # factor(+10 K) = exp(80/0.008314 * (1/298.15 - 1/308.15)); average = 0.9 * 1 + 0.1 * factor
    t = physics._Temperature(pd.DataFrame({"harsh": [0], "target_days": [10]}), 1, np.random.default_rng(0), CONSTANTS, np.array([[25.0]]))
    t.active, t.w, t.amp, t.W = True, np.array([[[0.1]]]), np.array([[[10.0]]]), np.array([[0.1]])
    factor = math.exp(80.0 / 8.314e-3 * (1 / 298.15 - 1 / 308.15))
    got = t.avg(lambda T: physics.arrhenius(80.0, T, 25.0))
    assert got[0, 0] == pytest.approx(0.9 + 0.1 * factor, rel=1e-9) and factor == pytest.approx(2.8499, rel=1e-3)


def test_excursions_shorten_life_and_harsh_transport_more():
    rows = [row("Potato Chips", rh_pct=75, wvtr38=2.0, harsh=h) for h in (0, 1)]
    v1, b4 = sim(rows, n=2048), sim(rows, n=2048, c=Constants(temp_history="excursions"))
    assert (b4["shelf_life_p50"] < v1["shelf_life_p50"]).all()
    drop = 1 - b4["shelf_life_p50"] / v1["shelf_life_p50"]
    assert drop.iloc[1] > drop.iloc[0]


# ------------------------------------------------------------------------------------------------ B5 transient produce gas

def test_o2_transient_matches_the_closed_form_when_respiration_is_zero_order():
    # Km -> 0 makes respiration constant R: V dx/dt = G (0.209 - x) - R  ->  x(t) = x_ss + (x0 - x_ss) exp(-G t / V), x_ss = 0.209 - R/G
    g, r, v, x0 = 50.0, 5.0, 1000.0, 0.209
    times, xs = o2_transient(x0, None, g, r, 1e-12, 1.0, v, np.array([[100.0]]), steps=20_000)
    x_ss = 0.209 - r / g
    expect = x_ss + (x0 - x_ss) * np.exp(-g * times[:, 0, 0] / v)
    assert np.allclose(xs[:, 0, 0], expect, atol=2e-4)


def test_transient_changes_nothing_without_ma_benefit_and_shortens_life_with_it():
    base = dict(temp_c=4, rh_pct=95, target_days=5, mass_kg=1.0, area_m2_per_kg=0.6, otr23=4000, wvtr38=10, seal=5, strength=4,
                perf_mm2_per_kg=0.5, fragility=0.0)
    no_ma = [row("Strawberry", ma_gain=0.0, **base)]
    assert sim(no_ma, c=Constants(produce_gas="transient"))["life_quality_p50"].iloc[0] == pytest.approx(sim(no_ma)["life_quality_p50"].iloc[0], rel=1e-6)
    ma = [row("Strawberry", ma_gain=1.0, **base)]
    steady = sim(ma)["life_quality_p50"].iloc[0]
    small = sim(ma, c=Constants(produce_gas="transient", produce_free_volume_cm3_per_kg=100.0))["life_quality_p50"].iloc[0]
    big = sim(ma, c=Constants(produce_gas="transient", produce_free_volume_cm3_per_kg=20000.0))["life_quality_p50"].iloc[0]
    assert big < small <= steady * (1 + 1e-9)                               # air at sealing: the benefit arrives late


# ------------------------------------------------------------------------------------------------ moisture at the target day (output only)

def test_v1_moisture_at_target_follows_the_linear_closed_form():
    # Banana Chips (iso_b 0.09), 50 g, 0.03 m2, WVTR38 2.0, 25 C, RH 70 %, start 2.6 % d.b. (moisture_pct 2.534113 w.b.), 120 days:
    # p_sat(25) 3.16767 kPa; Arrhenius(25 kJ, 25 vs 38 C) 0.656144; driving force at 38 C/90 % = 5.96209 kPa
    # C_wv = 2.0 x 0.656144 / 5.96209 x 0.03 x 3.16767 = 0.0209167 g/day; Ws = 50 x (1 - 0.02534113) = 48.7329 g
    # k = C_wv / (Ws b) = 0.0047690 /day; Me = 0.09 x 0.70 = 0.063
    # M(120) = 0.063 - (0.063 - 0.026) exp(-0.572280) = 0.042123 (4.21 % d.b.); aw = M / b = 0.46804
    r = sim([row("Banana Chips", temp_c=25, rh_pct=70, target_days=120, mass_kg=0.05, area_m2_per_kg=0.6, wvtr38=2.0,
                 moisture_pct=2.534113, seal=5)], c=Constants(**QUIET))
    assert r["moisture_db_at_target_p50"].iloc[0] == pytest.approx(0.042123, rel=2e-3)
    assert r["aw_at_target_p50"].iloc[0] == pytest.approx(0.46804, rel=2e-3)


def test_gab_aw_at_time_reduces_to_the_linear_solution_and_inverts_the_time_function():
    # C = 1, K = 1e-3, m0 = 90: M = 90 x 1e-3 aw / (1 - 1e-3 aw) ~ 0.09 aw (linear within 0.1 %).
    # Linear: aw(t) = RH - (RH - aw0) exp(-k t / b) = 0.7 - 0.5 exp(-0.0009 x 100 / 0.09) = 0.7 - 0.5 e^-1 = 0.516060
    aw = physics.aw_at_time_gab(0.2, 0.7, 0.0009, 100.0, 90.0, 1.0, 1e-3)
    assert float(aw) == pytest.approx(0.516060, abs=2e-3)
    # a real GAB shape: the time to reach the returned aw is the time asked for
    a_t = physics.aw_at_time_gab(0.15, 0.65, 0.002, 60.0, 0.05, 10.0, 0.9)
    assert float(time_to_moisture_gab(0.15, a_t, 0.65, 0.002, 0.05, 10.0, 0.9)) == pytest.approx(60.0, rel=1e-6)
    assert np.isnan(physics.aw_at_time_gab(0.7, 0.6, 0.002, 60.0, 0.05, 10.0, 0.9))        # drying is not modelled


# ------------------------------------------------------------------------------------------------ B1e empirical curve, valid aw range

def test_literature_and_empirical_gab_rows_are_kept_separate():
    # sorption_params.csv holds only an empirical-fit row for Banana Chips: "gab" must ignore it, "gab_empirical" must use it
    chips = row("Banana Chips", moisture_pct=2.534113, rh_pct=60, target_days=120, mass_kg=0.05, area_m2_per_kg=0.6, wvtr38=2.0)
    lit = sim([chips], c=Constants(isotherm="gab"))["sim_model"].iloc[0]
    emp = sim([chips], c=Constants(isotherm="gab_empirical"))["sim_model"].iloc[0]
    assert "fallback: no sourced GAB row" in lit and "iso=gab_empirical" in emp
    potato = sim([row("Potato Chips")], c=Constants(isotherm="gab_empirical"))["sim_model"].iloc[0]
    assert "fallback: no empirical GAB row" in potato


def test_curve_is_not_used_outside_its_valid_aw_range():
    # GAB m0 0.05, C 10, K 0.9 (aw 0.3 <-> M 0.053916, from the hand value above). Start at M0 = 0.053916 d.b. -> aw0 = 0.30.
    gab = dict(gab_m0=0.05, gab_C=10.0, gab_K=0.9, moisture_pct=100 * 0.053916 / 1.053916, rh_pct=70, target_days=120, wvtr38=5.0)
    base = row("Potato Chips", **gab)
    v1 = sim([base], c=Constants(**QUIET))
    free = sim([base], c=Constants(isotherm="gab", **QUIET))
    # 1) aw0 = 0.30 below valid_aw_min 0.35: every trial falls back to v1, life AND target-day state
    low = sim([base | dict(gab_aw_min=0.35, gab_aw_max=0.9)], c=Constants(isotherm="gab", **QUIET))
    for col in ("life_moisture_p50", "moisture_db_at_target_p50", "aw_at_target_p50", "aw0_p50"):
        assert low[col].iloc[0] == pytest.approx(v1[col].iloc[0])
    assert "life 100%, target-day state 100%" in low["sim_model"].iloc[0]
    # 2) range 0.25-0.95 covers aw0 0.30 -> RH 0.70: identical to the unbounded curve, nothing recorded
    wide = sim([base | dict(gab_aw_min=0.25, gab_aw_max=0.95)], c=Constants(isotherm="gab", **QUIET))
    for col in ("life_moisture_p50", "moisture_db_at_target_p50", "aw_at_target_p50"):
        assert wide[col].iloc[0] == pytest.approx(free[col].iloc[0])
    assert wide["sim_model"].iloc[0].startswith("iso=gab|")
    # 3) max 0.55: the critical aw (Potato Chips 0.45 < 0.55) is reached inside the range, so the life uses the curve, but the
    #    unbounded day-120 aw is above 0.55, so the target-day state falls back to v1
    assert free["aw_at_target_p50"].iloc[0] > 0.55
    mid = sim([base | dict(gab_aw_min=0.25, gab_aw_max=0.55)], c=Constants(isotherm="gab", **QUIET))
    assert mid["life_moisture_p50"].iloc[0] == pytest.approx(free["life_moisture_p50"].iloc[0])
    assert mid["aw_at_target_p50"].iloc[0] == pytest.approx(v1["aw_at_target_p50"].iloc[0])
    assert "life 0%, target-day state 100%" in mid["sim_model"].iloc[0]

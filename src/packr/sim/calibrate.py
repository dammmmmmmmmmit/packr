"""Calibration machinery (spec T3 part C). Scores the simulator against literature pairings; can fit at most 3 parameters.

    python -m packr.sim.calibrate --out reports

reference/literature_pairings.csv: one row per observed (food, film, conditions, end point).
  endpoint          "<quantity> <limit>": "moisture_db_pct 3.0" (% dry basis), "pv_meq_kg 10" (peroxide value), or
                    "texture_sensory" (the commodity's critical aw); observed_value is what the paper reported at observed_weeks
  outcome_pass      False = the end point was REACHED at observed_weeks; True = NOT reached by observed_weeks (right-censored)
  verified_by_human only verified rows count in the official score; the others are shown as provisional and are not evidence

Unknown inputs are never fixed to whatever fits best. Each is swept over the stated range below (and a stated temperature range
is swept end to end); the result for a pairing is the INTERVAL of the simulated quantity over that sweep. A verdict is
"consistent" / "inconsistent" only if the whole interval agrees, otherwise "undetermined".

Continuous score (`score_continuous`): rows that report moisture over time (`initial_moisture_db_pct` filled, endpoint not one of the
pass/fail kinds above) are scored on the predicted moisture (% d.b.) and aw at observed_weeks against `obs_moisture_db_pct` /
`obs_aw` (a number, or ">x" for a lower bound; blank = not yet verified). The run starts from the paper's measured initial moisture.
When `film_test_temp_c` is blank, the film's WVTR test temperature is unknown and swept over SWEEP_FILM_TEST_T_C; the WVTR is
converted to the simulator's 38 C / 90 % RH convention through the vapour-pressure difference and the simulator's own Ea for WVTR.

Fitting (`fit_lopo`): leave-one-pairing-out over the VERIFIED pairings, at most the 3 parameters ranked most sensitive in
reports/sensitivity.json for the category, each searched only inside a sourced prior range from reference/parameter_priors.csv
(rows without a doi are ignored). The objective uses the log error at the MIDPOINT of each sweep interval, so a nuisance input is
never chosen to fit. Nothing is written to commodities_master.csv or Constants: results are a report for approval.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..commodities import load_commodities
from ..paths import REFERENCE_DIR
from .physics import CONSTANTS, WVTR_DP_KPA, Constants, _sorption_table, arrhenius, gab_aw, gab_moisture, psat_kpa, simulate

PAIRINGS_CSV = REFERENCE_DIR / "literature_pairings.csv"
PRIORS_CSV = REFERENCE_DIR / "parameter_priors.csv"
SORPTION_POINTS_CSV = REFERENCE_DIR / "sorption_points.csv"
BINARY_KINDS = {"moisture_db_pct", "pv_meq_kg", "texture_sensory"}

# Stated plausible ranges for inputs a paper did not report (swept, never fitted)
SWEEP_RH_PCT = (30.0, 55.0, 80.0)                     # ambient storage with no RH given: dry to humid indoor
SWEEP_FILL_G_PER_600CM3 = (20.0, 40.0, 60.0)          # potato chips in a 600 cm3 bag (fill mass not reported)
SWEEP_COOKIE_MASS_G = (50.0, 100.0, 200.0)            # cookie pack mass not reported
SWEEP_COOKIE_AREA_M2_PER_KG = (0.8, 1.5, 2.5)         # cookie pack area not reported (range of the dataset's snack packs)
DEFAULT_SEAL, DEFAULT_STRENGTH = 4, 3                  # not reported: the v1 anchors' values
LIGHT_METALLISED, LIGHT_CLEAR = 0.95, 0.10            # light barrier by film description ("metal" in the name)
SWEEP_FILM_TEST_T_C = (23.0, 30.5, 38.0)              # WVTR test temperature not stated: the two common standard temperatures and between
FILM_TEST_RH = 0.90                                    # test RH not stated either: held at the simulator's own WVTR convention (not swept)
N_TRIALS = 512

MODELS = {
    "v1": Constants(),
    "B1 GAB isotherm": Constants(isotherm="gab"),
    "B2 PV oxidation": Constants(oxidation="pv"),
    "B3 RH-dependent OTR": Constants(rh_dependent_otr=True),
    "B4 temperature excursions": Constants(temp_history="excursions"),
    "B5 transient produce gas": Constants(produce_gas="transient"),
}


def load_pairings(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or PAIRINGS_CSV, keep_default_na=False)
    df["verified_by_human"] = df["verified_by_human"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["outcome_pass"] = df["outcome_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def _range(text: str, default: tuple) -> tuple:
    """'15-20' -> (15, 17.5, 20); '22' -> (22,); '' -> default. A trailing note in brackets ('20-30 (ambient)') is ignored."""
    text = str(text).split("(")[0].strip()
    if not text:
        return default
    if "-" in text[1:]:
        lo, hi = (float(x) for x in text.split("-", 1))
        return (lo, (lo + hi) / 2, hi)
    return (float(text),)


def scenario_rows(p: pd.Series, master: pd.DataFrame) -> pd.DataFrame:
    """Every combination of the swept inputs for one pairing, as simulator rows."""
    c = master.set_index("commodity").loc[p["commodity_in_table"]]
    temps = _range(p["temp_c"], ())
    rhs = _range(p["rh_pct"], SWEEP_RH_PCT)
    if str(p["pack_mass_g"]).strip():
        masses = (float(p["pack_mass_g"]),)
    elif str(p["pack_area_cm2"]).strip():
        masses = SWEEP_FILL_G_PER_600CM3
    else:
        masses = SWEEP_COOKIE_MASS_G
    areas = (None,) if str(p["pack_area_cm2"]).strip() else SWEEP_COOKIE_AREA_M2_PER_KG
    flush = 2.0 if "2 % O2" in str(p["atmosphere"]) else 20.9
    rows = []
    for t, rh, m, a in itertools.product(temps, rhs, masses, areas):
        r = c.to_dict()
        mass_kg = m / 1000.0
        area = float(p["pack_area_cm2"]) / 1e4 / mass_kg if a is None else a
        r.update(commodity=p["commodity_in_table"], category=c["category"], temp_c=t, rh_pct=rh, harsh=0,
                 target_days=float(p["observed_weeks"]) * 7.0, otr23=float(p["otr_cm3_m2_d"]), wvtr38=float(p["wvtr_g_m2_d"]),
                 light_barrier=LIGHT_METALLISED if "metal" in str(p["film"]).lower() else LIGHT_CLEAR, seal=DEFAULT_SEAL,
                 strength=DEFAULT_STRENGTH, mass_kg=mass_kg, area_m2_per_kg=area, perf_mm2_per_kg=0.0, flush_o2_pct=flush,
                 sweep=f"T={t:g} C, RH={rh:g} %, mass={m:g} g" + ("" if a is None else f", area={a:g} m2/kg"))
        rows.append(r)
    return pd.DataFrame(rows)


def endpoint(p) -> tuple[str, float | None]:
    parts = str(p["endpoint"]).split()
    return parts[0], (float(parts[1]) if len(parts) > 1 else None)


def is_binary(p) -> bool:
    return str(p["endpoint"]).split()[0] in BINARY_KINDS if str(p["endpoint"]).strip() else False


def is_continuous(p) -> bool:
    return not is_binary(p) and str(p.get("initial_moisture_db_pct", "")).strip() != ""


def score_pairing(p: pd.Series, c: Constants, master: pd.DataFrame, overrides: dict | None = None, seed: int = 0) -> dict:
    rows = scenario_rows(p, master)
    kind, limit = endpoint(p)
    if kind == "moisture_db_pct":
        rows["crit_moisture_db"] = limit / 100.0
    if kind == "pv_meq_kg" and limit is not None:
        c = dataclasses.replace(c, pv_endpoint_meq_kg=limit)
    for k, v in (overrides or {}).items():
        if k in rows.columns:
            rows[k] = v
    res = simulate(rows, n_trials=N_TRIALS, seed=seed, c=c, extra=True)
    target = float(p["observed_weeks"]) * 7.0
    if kind in ("moisture_db_pct", "texture_sensory"):
        life, prob = res["life_moisture_p50"].astype(float), res["p_moisture_by_target"].astype(float)
    else:
        life, prob = res["life_oxidation_p50"].astype(float), res["p_oxidation_by_target"].astype(float)
    reached = not p["outcome_pass"]
    if reached:
        logerr = np.log(np.clip(life, 0.05, None) / target)
    else:                                    # right-censored: only a predicted life SHORTER than the observation is an error
        logerr = np.minimum(np.log(np.clip(life, 0.05, None) / target), 0.0)
    p_lo, p_hi = float(prob.min()), float(prob.max())
    if reached:
        verdict = "consistent" if p_lo > 0.5 else ("inconsistent" if p_hi < 0.5 else "undetermined")
    else:
        verdict = "consistent" if p_hi < 0.5 else ("inconsistent" if p_lo > 0.5 else "undetermined")
    model_used = sorted(set(res["sim_model"])) if "sim_model" in res else ["v1"]
    out = dict(pairing_id=p["pairing_id"], endpoint=p["endpoint"], observed_weeks=float(p["observed_weeks"]), reached=reached,
               life_weeks_lo=float(life.min()) / 7, life_weeks_hi=float(life.max()) / 7, p_reached_lo=p_lo, p_reached_hi=p_hi,
               logerr_lo=float(logerr.min()), logerr_hi=float(logerr.max()), logerr_mid=float(0.5 * (logerr.min() + logerr.max())),
               verdict=verdict, n_sweep=len(rows), model_used=model_used)
    if kind == "pv_meq_kg":
        pv = res["pv_at_target_p50"].astype(float)
        out["pv_at_observed_lo"], out["pv_at_observed_hi"] = (float(pv.min()), float(pv.max())) if pv.notna().any() else (None, None)
    return out


def score_all(pairings: pd.DataFrame, models: dict | None = None, master: pd.DataFrame | None = None) -> pd.DataFrame:
    master = load_commodities() if master is None else master
    out = []
    for name, c in (models or MODELS).items():
        for _, p in pairings.iterrows():
            if is_binary(p):
                out.append(dict(model=name, verified=bool(p["verified_by_human"])) | score_pairing(p, c, master))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ continuous score

def wvtr38_equivalent(wvtr, test_t_c, test_rh: float = FILM_TEST_RH, c: Constants = CONSTANTS):
    """A WVTR measured at (test_t_c, test_rh, dry other side) restated in the simulator's 38 C / 90 % RH convention:
    permeance = WVTR / (p_sat(T_test) RH_test), moved to 38 C with the simulator's Arrhenius factor, times the 38 C / 90 % driving force."""
    return wvtr * WVTR_DP_KPA / (psat_kpa(test_t_c) * test_rh) / arrhenius(c.ea_wv_kj, test_t_c, 38.0)


def continuous_rows(p: pd.Series, master: pd.DataFrame, c: Constants = CONSTANTS) -> pd.DataFrame:
    """scenario_rows x the film test temperature sweep, starting from the paper's measured initial moisture."""
    base = scenario_rows(p, master)
    m0_db = float(p["initial_moisture_db_pct"]) / 100.0
    base["moisture_pct"] = 100.0 * m0_db / (1.0 + m0_db)                 # dry basis -> wet basis (the simulator's input)
    tests = _range(p.get("film_test_temp_c", ""), SWEEP_FILM_TEST_T_C)
    parts = []
    for tt in tests:
        r = base.copy()
        r["wvtr38"] = wvtr38_equivalent(float(p["wvtr_g_m2_d"]), tt, c=c)
        r["film_test_t"] = tt
        r["sweep"] = r["sweep"] + f", WVTR test T={tt:g} C"
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def _compare(obs: str, lo: float, hi: float, nd: int) -> str:
    """Observed value vs the predicted interval: a number, or '>x' / '<x' for a bound."""
    obs = str(obs).strip()
    if not obs:
        return ""
    if obs[0] in "<>":
        x = float(obs[1:])
        above_ok, below_ok = (lo > x, hi <= x) if obs[0] == ">" else (hi < x, lo >= x)
        return "consistent" if above_ok else ("inconsistent" if below_ok else "undetermined")
    x = float(obs)
    if x > hi:
        return f"under-predicted by {x - hi:.{nd}f} or more"
    if x < lo:
        return f"over-predicted by {lo - x:.{nd}f} or more"
    return "within interval"


def _fallback_share(texts) -> float:
    """Mean share of trials whose target-day state fell back to v1, from the simulator's sim_model text."""
    shares = [float(m.group(1)) / 100 if (m := re.search(r"target-day state (\d+)%", str(t))) else 0.0 for t in texts]
    return float(np.mean(shares)) if shares else 0.0


def score_continuous(p: pd.Series, c: Constants, master: pd.DataFrame, seed: int = 0, overrides: dict | None = None) -> dict:
    rows = continuous_rows(p, master, c)
    for k, v in (overrides or {}).items():
        rows[k] = v
    res = simulate(rows, n_trials=N_TRIALS, seed=seed, c=c, extra=True)
    m = res["moisture_db_at_target_p50"].astype(float).to_numpy() * 100.0
    aw = res["aw_at_target_p50"].astype(float).to_numpy()
    temps, rhs = _range(p["temp_c"], ()), _range(p["rh_pct"], SWEEP_RH_PCT)
    # nominal point: middle of the stated ambient ranges, WVTR taken at the simulator's own 38 C convention
    nom = ((rows["temp_c"] == temps[len(temps) // 2]) & (rows["rh_pct"] == rhs[len(rhs) // 2]) & (rows["film_test_t"] == 38.0)).to_numpy()
    model_used = sorted(set(res["sim_model"])) if "sim_model" in res else ["v1"]
    out = dict(pairing_id=p["pairing_id"], film=p["film"], wvtr=float(p["wvtr_g_m2_d"]), weeks=float(p["observed_weeks"]),
               m_lo=float(np.nanmin(m)), m_hi=float(np.nanmax(m)), m_nominal=float(m[nom][0]) if nom.any() else np.nan,
               aw_lo=float(np.nanmin(aw)), aw_hi=float(np.nanmax(aw)), aw_nominal=float(aw[nom][0]) if nom.any() else np.nan,
               obs_m=str(p.get("obs_moisture_db_pct", "")).strip(), obs_aw=str(p.get("obs_aw", "")).strip(), n_sweep=len(rows),
               model_used=model_used, gab_used=any("iso=gab" in u for u in model_used),
               v1_share=_fallback_share(res["sim_model"]) if "sim_model" in res else 1.0)
    out["m_check"] = _compare(out["obs_m"], out["m_lo"], out["m_hi"], 1)
    out["aw_check"] = _compare(out["obs_aw"], out["aw_lo"], out["aw_hi"], 2)
    return out


def score_continuous_all(pairings: pd.DataFrame, models: dict | None = None, master: pd.DataFrame | None = None) -> pd.DataFrame:
    master = load_commodities() if master is None else master
    models = models or {k: MODELS[k] for k in ("v1", "B1 GAB isotherm")}
    out = []
    for name, c in models.items():
        for _, p in pairings.iterrows():
            if is_continuous(p):
                out.append(dict(model=name, verified=bool(p["verified_by_human"])) | score_continuous(p, c, master))
    return pd.DataFrame(out)


def isotherm_checks(master: pd.DataFrame | None = None, path=None) -> pd.DataFrame:
    """aw implied at each measured (moisture, aw) point in reference/sorption_points.csv: v1 linear (aw = M / iso_b) and, when
    sorption_params.csv holds a sourced row for the commodity, the B1 GAB isotherm. Check points only: nothing is fitted."""
    master = load_commodities() if master is None else master
    pts = pd.read_csv(path or SORPTION_POINTS_CSV, keep_default_na=False)
    b = master.set_index("commodity")["iso_b"]
    tab = _sorption_table()
    lit, emp = tab[~tab["empirical"]], tab[tab["empirical"]]
    curve = lambda t, name, m: float(gab_aw(m, *t.loc[name, ["m0", "C", "K"]].astype(float))) if name in t.index else np.nan
    out = []
    for r in pts.itertuples():
        m, aw = float(r.moisture_db), float(r.aw)
        out.append(dict(commodity=r.commodity, moisture_db=m, aw_measured=aw, iso_b=float(b[r.commodity]), aw_v1=m / float(b[r.commodity]),
                        aw_b1=curve(lit, r.commodity, m), aw_b1_empirical=curve(emp, r.commodity, m), temp_c=r.temp_c, doi=r.doi))
    return pd.DataFrame(out)


def loo_films(pairings: pd.DataFrame, master: pd.DataFrame | None = None, commodity: str = "Banana Chips") -> pd.DataFrame:
    """B1 empirical isotherm scored leave-one-film-out: refit the GAB curve without film i (sorption_fit), then predict film i's
    day-120 moisture and aw with Constants(isotherm='gab_empirical'), the valid aw range being that of the remaining points.
    Within-paper consistency check: the kinetics, pack and conditions of film i come from the same paper as the curve."""
    from . import sorption_fit
    master = load_commodities() if master is None else master
    c = Constants(isotherm="gab_empirical")
    films = sorption_fit.points(pairings, commodity)
    out = []
    for pid in films.loc[films["label"] != "initial", "label"]:
        f = sorption_fit.fit_gab(sorption_fit.points(pairings, commodity, exclude=(pid,)))
        p = pairings.set_index("pairing_id").loc[pid].copy()
        p["pairing_id"] = pid
        ov = dict(gab_m0=f["m0"], gab_C=f["C"], gab_K=f["K"], gab_aw_min=f["aw_min"], gab_aw_max=f["aw_max"])
        out.append(dict(model="B1e leave-one-film-out", verified=bool(p["verified_by_human"])) | score_continuous(p, c, master, overrides=ov)
                   | dict(fold_m0=f["m0"], fold_C=f["C"], fold_K=f["K"], fold_aw_max=f["aw_max"]))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ fitting

def load_priors(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or PRIORS_CSV, keep_default_na=False)
    df = df[(df["doi"].astype(str).str.strip() != "") & (df["low"].astype(str) != "") & (df["high"].astype(str) != "")]
    return df.assign(low=pd.to_numeric(df["low"]), high=pd.to_numeric(df["high"]))


def top_sensitive(category: str, k: int = 3, path: Path = Path("reports/sensitivity.json")) -> list[str]:
    ranking = json.loads(path.read_text())[category]
    names = [r["factor"] for r in sorted(ranking, key=lambda r: r["rank_success"]) if not r.get("sourced")]
    return [n.replace("Constants.", "") for n in names[:k]]


def _apply_params(values: dict, base: Constants) -> tuple[Constants, dict]:
    const = {k: v for k, v in values.items() if k in {f.name for f in dataclasses.fields(Constants)}}
    rows = {k: v for k, v in values.items() if k not in const}
    return dataclasses.replace(base, **const), rows


def fit_lopo(pairings: pd.DataFrame, priors: pd.DataFrame, params: list[str], base: Constants = CONSTANTS, points: int = 7,
             master: pd.DataFrame | None = None) -> dict:
    """Leave-one-pairing-out grid search over sourced prior boxes. Returns a dict with status and, if run, per-fold results."""
    master = load_commodities() if master is None else master
    verified = pairings[pairings["verified_by_human"]].reset_index(drop=True)
    params = params[:3]
    missing = [p for p in params if p not in set(priors["parameter"])]
    if len(verified) < 3:
        return dict(status="not run", reason=f"{len(verified)} verified pairings (need at least 3 for leave-one-out)")
    if missing:
        return dict(status="not run", reason=f"no sourced prior range for {missing}")
    box = {p: np.linspace(*priors.set_index("parameter").loc[p, ["low", "high"]].astype(float), points) for p in params}
    grid = [dict(zip(box, vals)) for vals in itertools.product(*box.values())]
    errs = np.zeros((len(grid), len(verified)))
    for gi, values in enumerate(grid):
        c, rows = _apply_params(values, base)
        for pi, p in verified.iterrows():
            errs[gi, pi] = score_pairing(p, c, master, overrides=rows)["logerr_mid"]
    folds = []
    for held in range(len(verified)):
        train = [i for i in range(len(verified)) if i != held]
        best = int(np.argmin((errs[:, train] ** 2).sum(1)))
        folds.append(dict(held_out=verified.loc[held, "pairing_id"], fitted=grid[best], train_rmse=float(np.sqrt((errs[best, train] ** 2).mean())),
                          held_out_logerr=float(errs[best, held]), v1_held_out_logerr=None))
    return dict(status="run", params=params, prior_box={k: [float(v[0]), float(v[-1])] for k, v in box.items()}, folds=folds)


# ------------------------------------------------------------------------------------------------ report

def _fmt_iv(lo, hi, nd=2):
    return f"{lo:.{nd}f}" if abs(hi - lo) < 10 ** -nd else f"{lo:.{nd}f} to {hi:.{nd}f}"


def _continuous_section(cont: pd.DataFrame, iso: pd.DataFrame, pairings: pd.DataFrame, master: pd.DataFrame,
                        loo: pd.DataFrame | None = None, sfit: dict | None = None) -> list[str]:
    if cont is None or cont.empty:
        return []
    v1 = cont[cont["model"] == "v1"]
    p0 = pairings.set_index("pairing_id").loc[v1["pairing_id"].iloc[0]]
    commodity = p0["commodity_in_table"]
    b = float(master.set_index("commodity").loc[commodity, "iso_b"])
    rh_max = max(_range(p0["rh_pct"], SWEEP_RH_PCT))
    f23 = float(wvtr38_equivalent(1.0, 23.0))
    lines = ["## Moisture over time (continuous score)", "",
             f"Rows that report moisture and aw at a fixed time: {v1['pairing_id'].iloc[0]}..{v1['pairing_id'].iloc[-1]} "
             f"({len(v1)} films, {p0['source'].split(',')[0]}, doi {p0['doi']}), all unverified. Each run starts from the measured initial moisture "
             f"({p0['initial_moisture_db_pct']} % d.b.) with the stated pack ({p0['pack_mass_g']} g, {p0['pack_area_cm2']} cm2). Swept: storage temperature "
             f"{p0['temp_c']} C and RH {p0['rh_pct']} % (each end to end, held constant over the run), and the film's WVTR test temperature "
             + ", ".join(f"{t:g}" for t in SWEEP_FILM_TEST_T_C) + f" C (not stated in the paper). A WVTR measured at 23 C is restated as {f23:.2f}x the same "
             f"number measured at 38 C (vapour pressure and the simulator's Ea for WVTR); the test RH is held at {FILM_TEST_RH:.0%} (also not stated, not swept: "
             "a lower test RH would widen the intervals upward). Values are trial medians at day "
             f"{float(p0['observed_weeks']) * 7:.0f}; the interval is over the {int(v1['n_sweep'].iloc[0])} sweep points; *nominal* = mid temperature, mid RH, "
             "WVTR taken at 38 C.", "",
             "Observed values: all 15 films, read from the paper's tables by an automated reader and cross-checked against its text, NOT checked "
             "against the PDF (verified_by_human = False). Blank = not reported. "
             f"Under v1 the moisture can never exceed the linear-isotherm equilibrium iso_b x RH = {b} x {rh_max / 100:.3f} = {100 * b * rh_max / 100:.1f} % d.b. "
             "at the highest stated RH (plus the RH noise of the draws), whatever the film.", "",
             "| Film | WVTR (as reported) | v1 moisture % d.b. (interval) | nominal | observed | moisture check | v1 aw (interval) | nominal | observed | aw check |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in v1.itertuples():
        lines.append(f"| {r.pairing_id} ({r.film}) | {r.wvtr:g} | {r.m_lo:.1f} to {r.m_hi:.1f} | {r.m_nominal:.1f} | {r.obs_m} | {r.m_check} | "
                     f"{r.aw_lo:.2f} to {r.aw_hi:.2f} | {r.aw_nominal:.2f} | {r.obs_aw} | {r.aw_check} |")
    b1 = cont[cont["model"] == "B1 GAB isotherm"]
    lines += ["", "Moisture and aw are checked separately. v1's linear isotherm ties them (aw = M / iso_b), so a film can pass both checks "
              "with no single sweep point matching both: compare the check points below."]
    lines.append("")
    lines += _empirical_section(cont, loo, sfit, v1, master)
    if not b1.empty and b1["gab_used"].any():
        lines += ["### B1 GAB isotherm", "", "| Film | moisture % d.b. (interval) | observed | check | aw (interval) | observed | check |", "|---|---|---|---|---|---|---|"]
        for r in b1.itertuples():
            lines.append(f"| {r.pairing_id} | {r.m_lo:.1f} to {r.m_hi:.1f} | {r.obs_m} | {r.m_check} | {r.aw_lo:.2f} to {r.aw_hi:.2f} | {r.obs_aw} | {r.aw_check} |")
        lines.append("")
    else:
        lines += [f"B1 GAB isotherm (literature constants): not reported, `sorption_params.csv` has no literature row for {commodity} "
                  "(B1 would equal v1).", ""]
    if iso is not None and not iso.empty:
        lines += ["### Isotherm check points (`reference/sorption_points.csv`)", "",
                  "Measured (moisture, aw) pairs. Two points cannot identify the three GAB constants; the empirical column is IN-SAMPLE (both points "
                  "are among the 16 the curve was fitted to).", "",
                  "| Commodity | Moisture (d.b.) | aw measured | v1 aw = M / iso_b | B1 literature GAB aw | B1e empirical curve aw (in-sample) | Temperature |",
                  "|---|---|---|---|---|---|---|"]
        for r in iso.itertuples():
            gab = "no literature row" if np.isnan(r.aw_b1) else f"{r.aw_b1:.2f}"
            emp = "no empirical row" if np.isnan(r.aw_b1_empirical) else f"{r.aw_b1_empirical:.2f}"
            lines.append(f"| {r.commodity} | {r.moisture_db:.3f} | {r.aw_measured:.2f} | {r.aw_v1:.2f} (iso_b {r.iso_b:g}) | {gab} | {emp} | "
                         f"{r.temp_c or 'not recorded'} |")
        lines.append("")
    return lines


def _num(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _summary(g: pd.DataFrame) -> dict:
    om, oa = g["obs_m"].map(_num), g["obs_aw"].map(_num)
    em, ea = g["m_nominal"] - om, g["aw_nominal"] - oa
    return dict(n_m=int(om.notna().sum()), in_m=int((g["m_check"] == "within interval").sum()), under_m=int(g["m_check"].str.startswith("under").sum()),
                rmse_m=float(np.sqrt(np.nanmean(em ** 2))), bias_m=float(np.nanmean(em)), n_aw=int(oa.notna().sum()),
                in_aw=int((g["aw_check"] == "within interval").sum()), rmse_aw=float(np.sqrt(np.nanmean(ea ** 2))), bias_aw=float(np.nanmean(ea)))


def _empirical_section(cont: pd.DataFrame, loo: pd.DataFrame | None, sfit: dict | None, v1: pd.DataFrame, master: pd.DataFrame) -> list[str]:
    if sfit is None or loo is None or loo.empty:
        return []
    from . import sorption_fit
    f, boot, ident, pts = sfit["fit"], sfit["boot"], sfit["ident"], sfit["points"]
    q = boot.quantile([0.025, 0.5, 0.975])
    commodity = sfit["commodity"]
    fdc_wb = float(master.set_index("commodity").loc[commodity, "moisture_pct"])
    fdc_db = fdc_wb / (100.0 - fdc_wb)
    aw_fdc, aw_fdc_b = float(sorption_fit.aw_at(fdc_db, f)), sorption_fit.aw_at(fdc_db, boot)
    unw = sorption_fit.fit_gab(pts, weighted=False)
    n_init = int((pts["sd_source"] != "reported").sum())
    lines = ["## B1e: empirical GAB curve for Banana Chips (`isotherm=\"gab_empirical\"`)", "",
             f"Fitted by `python -m packr.sim.sorption_fit` to {len(pts)} (moisture d.b., aw) pairs from the same paper: the initial point "
             f"(2.6 % d.b., aw 0.16) and the 15 day-120 film pouches, all **unverified against the PDF**. Weights: effective variance from the "
             f"reported SDs of both moisture and aw; the {n_init} point without SDs (initial) gets the median reported SDs "
             f"({pts['sd_m'].median() * 100:.2f} pp, aw {pts['sd_aw'].median():.3f}). chi2 {f['chi2']:.1f} on {f['dof']} dof. "
             f"Bootstrap: {len(boot)} resamples of the 15 films, initial point always kept. Search bounds: "
             + ", ".join(f"{k} {lo:g} to {hi:g}" for k, (lo, hi) in sorption_fit.BOUNDS.items()) + ".", "",
             "| Constant | Fit | Bootstrap 2.5 % | median | 97.5 % |", "|---|---|---|---|---|"]
    for k in ("m0", "C", "K"):
        lines.append(f"| {k} | {f[k]:.4g} | {q.loc[0.025, k]:.4g} | {q.loc[0.5, k]:.4g} | {q.loc[0.975, k]:.4g} |")
    verdict = ("**The constants are not identifiable**: " + "; ".join(ident["reasons"]) + ". Stored in `sorption_params.csv` with trust "
               f"`{sorption_fit.TRUST_UNIDENTIFIABLE}`, valid aw {f['aw_min']:g} to {f['aw_max']:g}: only the curve is meant to be used, never "
               "the individual constants." if not ident["identifiable"] else "The constants are identifiable by the stated rule.")
    lines += ["", verdict, "", "The curve itself is better determined than its constants (moisture d.b. %, bootstrap 2.5 to 97.5 %):", "",
              "| aw | " + " | ".join(f"{a:g}" for a in (0.16, 0.3, 0.4, 0.5, 0.6, 0.7)) + " |", "|---" * 7 + "|",
              "| curve | " + " | ".join(f"{100 * float(sorption_fit.moisture_at(a, f)):.2f} ({100 * np.percentile(sorption_fit.moisture_at(a, boot), 2.5):.2f} to "
                                     f"{100 * np.percentile(sorption_fit.moisture_at(a, boot), 97.5):.2f})" for a in (0.16, 0.3, 0.4, 0.5, 0.6, 0.7)) + " |",
              "", f"**aw at the FDC moisture of Banana Chips ({fdc_wb:g} % w.b. = {100 * fdc_db:.2f} % d.b.): {aw_fdc:.2f}** (bootstrap "
              f"{np.percentile(aw_fdc_b, 2.5):.2f} to {np.percentile(aw_fdc_b, 97.5):.2f}). An unweighted fit (moisture residuals, unit weights) gives "
              f"{float(sorption_fit.aw_at(fdc_db, unw)):.2f}: the difference is the weighting, not the data. v1 (iso_b "
              f"{float(master.set_index('commodity').loc[commodity, 'iso_b']):g}) gives {fdc_db / float(master.set_index('commodity').loc[commodity, 'iso_b']):.2f}; "
              f"crit_aw stays {float(master.set_index('commodity').loc[commodity, 'crit_aw']):g} (not changed).", ""]
    ins = cont[cont["model"] == "B1e empirical (in-sample)"]
    s_v1, s_loo = _summary(v1), _summary(loo)
    lines += ["### Leave-one-film-out (within-paper consistency check, NOT independent validation)", "",
              "For each film i the curve is refitted without film i (15 points), then film i's day-120 moisture and aw are predicted with "
              "`isotherm=\"gab_empirical\"` and the same sweep as v1 (storage T, RH, WVTR test temperature). The curve, the kinetics inputs and the "
              "observations all come from one paper, so agreement here shows internal consistency only. Outside the curve's valid aw range "
              "(that of the 15 remaining points) a trial falls back to v1; *v1 share* is the mean share of trials that did so over the sweep "
              "(the high-RH sweep points carry most of it, which pulls the top of the B1e interval toward v1).", "",
              f"Summary (nominal = mid T, mid RH, WVTR at 38 C): moisture within interval v1 {s_v1['in_m']}/{s_v1['n_m']}, B1e-LOO {s_loo['in_m']}/{s_loo['n_m']}; "
              f"under-predicted v1 {s_v1['under_m']}, B1e-LOO {s_loo['under_m']}; nominal RMSE v1 {s_v1['rmse_m']:.2f} pp (bias {s_v1['bias_m']:+.2f}), "
              f"B1e-LOO {s_loo['rmse_m']:.2f} pp (bias {s_loo['bias_m']:+.2f}). aw within interval v1 {s_v1['in_aw']}/{s_v1['n_aw']}, B1e-LOO "
              f"{s_loo['in_aw']}/{s_loo['n_aw']}; nominal aw RMSE v1 {s_v1['rmse_aw']:.3f} (bias {s_v1['bias_aw']:+.3f}), B1e-LOO {s_loo['rmse_aw']:.3f} "
              f"(bias {s_loo['bias_aw']:+.3f})."
              + (f" For reference only, the in-sample curve (all 16 points): moisture RMSE {_summary(ins)['rmse_m']:.2f} pp, "
                 f"{_summary(ins)['in_m']}/{_summary(ins)['n_m']} within interval." if not ins.empty else ""), "",
              "| Film | Obs. moisture % d.b. (SD) | v1 nominal (interval) | v1 check | B1e-LOO nominal (interval) | B1e-LOO check | fold curve at obs. aw | v1 share |",
              "|---|---|---|---|---|---|---|---|"]
    v1i = v1.set_index("pairing_id")
    for r in loo.itertuples():
        a = v1i.loc[r.pairing_id]
        sd = pts.set_index("label").loc[r.pairing_id, "sd_m"] * 100
        at_obs = 100 * float(gab_moisture(_num(r.obs_aw), r.fold_m0, r.fold_C, r.fold_K))
        lines.append(f"| {r.pairing_id} | {r.obs_m} ({sd:.2f}) | {a.m_nominal:.1f} ({a.m_lo:.1f} to {a.m_hi:.1f}) | {a.m_check} | "
                     f"{r.m_nominal:.1f} ({r.m_lo:.1f} to {r.m_hi:.1f}) | {r.m_check} | {at_obs:.1f} | {r.v1_share:.0%} |")
    lines += ["", "*fold curve at obs. aw* = the moisture the curve fitted without film i gives at film i's MEASURED aw: if it is far from the "
              "observed moisture, the film's own (moisture, aw) pair disagrees with the other films and no kinetics can fix it.", "",
              "| Film | Obs. aw (SD) | v1 nominal (interval) | v1 check | B1e-LOO nominal (interval) | B1e-LOO check |", "|---|---|---|---|---|---|"]
    for r in loo.itertuples():
        a = v1i.loc[r.pairing_id]
        sd = pts.set_index("label").loc[r.pairing_id, "sd_aw"]
        lines.append(f"| {r.pairing_id} | {r.obs_aw} ({sd:.3f}) | {a.aw_nominal:.2f} ({a.aw_lo:.2f} to {a.aw_hi:.2f}) | {a.aw_check} | "
                     f"{r.aw_nominal:.2f} ({r.aw_lo:.2f} to {r.aw_hi:.2f}) | {r.aw_check} |")
    lines.append("")
    return lines


def report(scores: pd.DataFrame, pairings: pd.DataFrame, fits: dict, cont: pd.DataFrame | None = None, iso: pd.DataFrame | None = None,
           master: pd.DataFrame | None = None, loo: pd.DataFrame | None = None, sfit: dict | None = None) -> str:
    n_ver = int(pairings["verified_by_human"].sum())
    lines = ["# Calibration against literature pairings", "",
             "Generated by `python -m packr.sim.calibrate` from `reference/literature_pairings.csv`. "
             f"**{n_ver} of {len(pairings)} pairings are verified by a human**; only verified rows are evidence. Unverified rows are shown "
             "below as *provisional* so the machinery and the structural options can be compared, not to judge the simulator.", "",
             "Unknown inputs are swept, never fitted: RH " + ", ".join(f"{v:g}" for v in SWEEP_RH_PCT) + " % when not reported; chips fill mass "
             + ", ".join(f"{v:g}" for v in SWEEP_FILL_G_PER_600CM3) + " g per 600 cm3 bag; cookie pack mass " + ", ".join(f"{v:g}" for v in SWEEP_COOKIE_MASS_G)
             + " g and area " + ", ".join(f"{v:g}" for v in SWEEP_COOKIE_AREA_M2_PER_KG) + " m2/kg; a stated temperature range is swept end to end. "
             f"Seal {DEFAULT_SEAL}/5 and strength {DEFAULT_STRENGTH}/5 (not reported). Light barrier {LIGHT_METALLISED} for metallised films, {LIGHT_CLEAR} otherwise. "
             f"{N_TRIALS} trials per sweep point.", "",
             "Columns: simulated median time to the end point (weeks) and P(end point reached by the observed week), each as the interval over the sweep; "
             "verdict = consistent / inconsistent only if the whole interval agrees with the observation; log error = ln(simulated / observed) "
             "(right-censored rows: only a shorter simulated life counts).", ""]
    official = scores[scores["verified"]]
    lines += ["## Official score (verified pairings)", ""]
    if official.empty:
        lines += ["No verified pairings yet: **no official score**. Verify rows (and fill in DOI, RH, fill mass) to make this section count.", ""]
    else:
        for m, g in official.groupby("model", sort=False):
            lines.append(f"* {m}: {(g['verdict'] == 'consistent').sum()} consistent, {(g['verdict'] == 'inconsistent').sum()} inconsistent, "
                         f"{(g['verdict'] == 'undetermined').sum()} undetermined; median |log error| at sweep midpoint {g['logerr_mid'].abs().median():.2f}")
        lines.append("")
    lines += ["## Provisional (unverified pairings)", "",
              "| Model | Directions consistent / inconsistent / undetermined | Median abs log error (midpoint) | Fell back to v1 on |", "|---|---|---|---|"]
    for m, g in scores.groupby("model", sort=False):
        used = "; ".join(sorted({u for us in g["model_used"] for u in us}))
        if "fallback" in used or "no sourced" in used:
            fb = "all rows (no sourced table rows: identical to v1)"
        elif m.startswith("B5") and not (g["endpoint"].str.startswith("produce")).any():
            fb = "not applicable (no produce pairings)"
        else:
            fb = "none"
        lines.append(f"| {m} | {(g['verdict'] == 'consistent').sum()} / {(g['verdict'] == 'inconsistent').sum()} / {(g['verdict'] == 'undetermined').sum()} | "
                     f"{g['logerr_mid'].abs().median():.2f} | {fb} |")
    lines += [""]
    for pid, g in scores.groupby("pairing_id", sort=False):
        p = pairings.set_index("pairing_id").loc[pid]
        obs = f"end point {'REACHED' if not p['outcome_pass'] else 'not reached'} at {p['observed_weeks']} weeks ({p['endpoint']} {p['observed_value']})"
        lines += [f"### {pid}: {p['food_description']}, {p['film']} (OTR {p['otr_cm3_m2_d']}, WVTR {p['wvtr_g_m2_d']}), {p['temp_c']} C, RH {p['rh_pct'] or 'not reported'}", "",
                  f"Observed: {obs}. {p['quality_notes']}", "",
                  "| Model | Sim. time to end point (weeks) | P(reached by observed week) | Verdict | Log error |", "|---|---|---|---|---|"]
        for r in g.itertuples():
            extra = f"; PV at week {p['observed_weeks']}: {_fmt_iv(r.pv_at_observed_lo, r.pv_at_observed_hi)}" if getattr(r, "pv_at_observed_lo", None) is not None and r.pv_at_observed_lo == r.pv_at_observed_lo else ""
            lines.append(f"| {r.model} | {_fmt_iv(r.life_weeks_lo, r.life_weeks_hi, 1)} | {_fmt_iv(r.p_reached_lo, r.p_reached_hi)} | {r.verdict} | "
                         f"{_fmt_iv(r.logerr_lo, r.logerr_hi)}{extra} |")
        lines.append("")
    lines += _continuous_section(cont, iso, pairings, load_commodities() if master is None else master, loo, sfit)
    lines += ["## Parameter fitting (leave one pairing out)", ""]
    for cat, f in fits.items():
        if f["status"] != "run":
            lines.append(f"* {cat}: **not run** ({f['reason']}). Candidate parameters (top 3 by sensitivity): {', '.join(f.get('candidates', []))}.")
        else:
            lines.append(f"* {cat}: parameters {f['params']} within sourced priors {f['prior_box']}")
            for fold in f["folds"]:
                lines.append(f"    * held out {fold['held_out']}: fitted {fold['fitted']}, train RMSE {fold['train_rmse']:.2f}, held-out log error {fold['held_out_logerr']:.2f}")
    lines += ["", "Nothing here changes `commodities_master.csv` or `Constants`; any fitted value needs approval and a citation for its prior."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="packr.sim.calibrate")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--sensitivity", default="reports/sensitivity.json")
    a = ap.parse_args(argv)
    pairings = load_pairings()
    master = load_commodities()
    scores = score_all(pairings, master=master)
    from . import sorption_fit
    cont = score_continuous_all(pairings, master=master, models={"v1": MODELS["v1"], "B1 GAB isotherm": MODELS["B1 GAB isotherm"],
                                                                 "B1e empirical (in-sample)": Constants(isotherm="gab_empirical")})
    iso = isotherm_checks(master)
    sfit = sorption_fit.run(pairings)
    loo = loo_films(pairings, master)
    priors = load_priors()
    fits = {}
    for cat in ("dry_snack",):
        cand = top_sensitive(cat, path=Path(a.sensitivity)) if Path(a.sensitivity).exists() else []
        cats = pairings[pairings["commodity_in_table"].isin(master.query("category == @cat")["commodity"]) & pairings.apply(is_binary, axis=1)]
        fits[cat] = fit_lopo(cats, priors, cand) | dict(candidates=cand)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.md").write_text(report(scores, pairings, fits, cont, iso, master, loo, sfit), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(dict(scores=scores.to_dict("records"), continuous=cont.to_dict("records"),
                                                          isotherm_checks=iso.to_dict("records"), leave_one_film_out=loo.to_dict("records"),
                                                          empirical_isotherm=dict(fit=sfit["fit"], ident=sfit["ident"],
                                                                                  bootstrap_quantiles=sfit["boot"].quantile([0.025, 0.5, 0.975]).to_dict()),
                                                          fits=fits), indent=2, default=str))
    v1 = scores[scores["model"] == "v1"]
    print(v1[["pairing_id", "verified", "life_weeks_lo", "life_weeks_hi", "p_reached_lo", "p_reached_hi", "verdict"]].round(2).to_string(index=False))
    cv1 = cont[cont["model"] == "v1"]
    print(cv1[["pairing_id", "wvtr", "m_lo", "m_hi", "m_nominal", "obs_m", "m_check", "aw_lo", "aw_hi", "obs_aw", "aw_check"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

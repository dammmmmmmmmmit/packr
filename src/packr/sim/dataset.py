"""Phase 4: synthetic dataset generation, grouped train / test split, literature anchors.

    python -m packr.sim.dataset --out data/datasets --seed 0

Row = one (scenario, commodity, package) triple, labelled by the Monte Carlo simulation in `packr.sim.physics`.

Leakage rules (enforced here and tested):
  * the split is by COMMODITY. A commodity, and every virtual copy made from it, is entirely in train or entirely in test.
  * virtual commodities are made only from TRAIN commodities. The test set uses real commodities at nominal parameters only.
  * the 5 cross-validation folds are also by commodity (a virtual copy inherits its parent's fold).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..commodities import MASTER_CSV, SUPPLIED_SHELF_LIFE, load_commodities
from ..engine.models import Scenario
from ..engine.risks import assess_risks
from ..engine.rules import derive_requirements
from ..materials import load_materials
from ..engine.structures import build_candidates
from .physics import COMMODITY_COLS, OUT_COLS, simulate

# Held-out real commodities. Chosen to span respiration classes and sub-groups, and fixed (not random) so the test set
# is reproducible and inspectable. Potato Chips stay in TRAIN because they are the headline use case; a final model should
# be retrained on every commodity after evaluation.
TEST_COMMODITIES = ["Strawberry", "Lettuce", "Mango", "Potato", "Cucumber", "Apple", "Okra", "Cauliflower",
                    "Banana Chips", "Cream Biscuits", "Roasted Peanuts"]
N_FOLDS = 5


# --------------------------------------------------------------------------- packaging candidates

def candidate_table(materials: pd.DataFrame) -> pd.DataFrame:
    """All rule-engine candidate structures with the sums needed to rescale bulk-layer gauge quickly."""
    rows = []
    for layers in build_candidates(materials):
        m = [materials.loc[n] for n in layers]
        bulk = np.array([r["barrier_type"] == "bulk" for r in m])
        inv_o = np.array([1.0 / float(r["OTR"]) for r in m])
        inv_w = np.array([1.0 / float(r["WVTR"]) for r in m])
        g = np.array([float(r["gauge_um"]) for r in m])
        rows.append(dict(
            layers="/".join(layers), n_layers=len(layers),
            inv_o_bulk=inv_o[bulk].sum(), inv_o_fixed=inv_o[~bulk].sum(),
            inv_w_bulk=inv_w[bulk].sum(), inv_w_fixed=inv_w[~bulk].sum(),
            gauge_bulk=g[bulk].sum(), gauge_fixed=g[~bulk].sum(),
            light=max(float(r["light"]) for r in m), seal=float(m[-1]["seal"]),
            strength=float(np.mean([r["strength"] for r in m])), cost=float(sum(r["cost"] for r in m)),
            cost_bulk=float(sum(r["cost"] for r, b in zip(m, bulk) if b)), cost_fixed=float(sum(r["cost"] for r, b in zip(m, bulk) if not b)),
            recyclable=len({r["family"] for r in m}) == 1, breathable=any(bool(r["breathable"]) for r in m)))
    return pd.DataFrame(rows)


def scaled(cands: pd.DataFrame, idx: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    c = cands.iloc[idx].reset_index(drop=True)
    return pd.DataFrame(dict(
        otr23=1.0 / (s * c["inv_o_bulk"] + c["inv_o_fixed"]), wvtr38=1.0 / (s * c["inv_w_bulk"] + c["inv_w_fixed"]),
        thickness_um=s * c["gauge_bulk"] + c["gauge_fixed"]))


def perforation_cm2_for_o2(otr23, area_m2_per_kg, mass_kg, temp_c, resp_at_t, x_o2):
    """Perforation area (cm2 per pack) that puts the steady-state in-pack O2 fraction near `x_o2`, from the film's O2 conductance
    and a Michaelis-Menten respiration (Km 2 % O2). Shared by the dataset design and the optimiser's search."""
    otr_t = otr23 * np.exp(-40.0 / 8.314e-3 * (1 / (temp_c + 273.15) - 1 / 296.15))
    g_film = otr_t * area_m2_per_kg * mass_kg
    r_air = resp_at_t * mass_kg * 13.1
    r_at_x = r_air * (x_o2 / (0.02 + x_o2)) / (0.209 / (0.02 + 0.209))
    need = r_at_x / (0.209 - x_o2)
    return np.clip(need - g_film, 0, None) / (0.2 * 86400.0 / 0.0025)


# --------------------------------------------------------------------------- commodities

_LOGN = ["resp_ref_mgco2_kg_h", "base_life_days", "o2_min_pct", "co2_max_pct", "transp_g_kg_d", "iso_b", "ox_ref_days", "fat_pct"]
_ADD = {"ideal_temp_c": 1.5, "crit_weight_loss_pct": 1.0, "ma_gain": 0.15, "q10_resp": 0.25, "q10_decay": 0.15, "crit_aw": 0.04,
        "ea_ox_kj": 4.0, "light_sens": 0.1, "ph": 0.3, "moisture_pct": 2.5, "fragility": 0.15}


def make_virtual(parent: pd.Series, k: int, rng: np.random.Generator) -> pd.Series:
    """A plausible sibling of a real commodity: continuous parameters jittered, categorical ones kept."""
    v = parent.copy()
    for col in _LOGN:
        if pd.notna(v[col]) and v[col] > 0:
            v[col] = v[col] * math.exp(rng.normal(0, 0.2 if col != "resp_ref_mgco2_kg_h" else 0.3))
    for col, sd in _ADD.items():
        if pd.notna(v[col]):
            v[col] = v[col] + rng.normal(0, sd)
    v["moisture_pct"] = float(np.clip(v["moisture_pct"], 0.5, 97))
    v["fat_pct"] = float(np.clip(v["fat_pct"], 0.1, min(60, 99 - v["moisture_pct"])))
    v["ph"] = float(np.clip(v["ph"], 2.5, 8))
    for col in ("q10_resp", "q10_decay"):
        if pd.notna(v[col]):
            v[col] = float(np.clip(v[col], 1.5, 3.2))
    for col in ("crit_aw", "ma_gain", "light_sens", "fragility"):
        if pd.notna(v[col]):
            v[col] = float(np.clip(v[col], 0.05, 0.95 if col in ("crit_aw",) else 2.0))
    if pd.notna(v["o2_min_pct"]):
        v["o2_min_pct"] = float(np.clip(v["o2_min_pct"], 0.3, 6))
    v["commodity"] = f"{parent['commodity']}~v{k}"
    v["commodity_id"] = f"{parent['commodity_id']}v{k:02d}"
    v["source"] = "virtual"
    return v


# --------------------------------------------------------------------------- scenario / pack sampling

def _sample_rows(c: pd.Series, n: int, rng: np.random.Generator, cands: pd.DataFrame) -> pd.DataFrame:
    snack = c["category"] == "dry_snack"
    chilled = rng.random(n) < (0.10 if snack else 0.65)
    temp = np.where(chilled, rng.uniform(0, 12, n), rng.uniform(15, 38, n))
    if snack:
        rh = rng.uniform(30, 90, n)
        target = np.exp(rng.uniform(math.log(30), math.log(400), n))
    else:
        rh = np.where(chilled, rng.uniform(80, 98, n), rng.uniform(50, 95, n))
        # Targets are drawn around the life the commodity can plausibly reach at that temperature (so labels are not all 0),
        # with 15 % wide-range targets so impossible requests are represented too.
        achievable = c["base_life_days"] * c["q10_decay"] ** (-(temp - c["ideal_temp_c"]) / 10.0)
        spread = np.where(rng.random(n) < 0.85, rng.uniform(math.log(0.3), math.log(1.6), n), rng.uniform(math.log(0.1), math.log(6.0), n))
        target = np.clip(achievable * np.exp(spread), 2, 365)
    harsh = rng.random(n) < 0.3
    mass = np.exp(rng.uniform(math.log(0.05), math.log(0.5), n)) if snack else np.exp(rng.uniform(math.log(0.25), math.log(5), n))
    area = np.exp(rng.uniform(math.log(0.6), math.log(2.5), n)) if snack else np.exp(rng.uniform(math.log(0.25), math.log(3), n))
    perf = np.zeros(n) if snack else np.where(rng.random(n) < 0.5, 0.0, np.exp(rng.uniform(math.log(0.01), math.log(30), n)))
    flush = np.where(rng.random(n) < 0.5, 20.9, rng.uniform(0.5, 5, n)) if snack else np.full(n, 20.9)
    resp_at_t = c["resp_ref_mgco2_kg_h"] * c["q10_resp"] ** ((temp - c["resp_ref_temp_c"]) / 10.0)
    # requirements from the rule engine, used (a) to bias half the structures toward feasible ones, (b) as the rules-only baseline
    req = []
    for i in range(n):
        s = Scenario(commodity=c["commodity"], moisture_pct=float(c["moisture_pct"]), fat_pct=float(c["fat_pct"]), ph=float(c["ph"]),
                     shelf_life_days=float(target[i]), temp_c=float(temp[i]), rh_pct=float(rh[i]),
                     storage="Chilled" if chilled[i] else "Ambient", transport="Harsh" if harsh[i] else "Normal",
                     respiration_mgco2_kg_h=float(resp_at_t[i]))
        req.append(derive_requirements(s, assess_risks(s)))
    r_otr_max = np.array([r.otr_max for r in req]); r_otr_min = np.array([r.otr_min for r in req])
    r_wvtr_max = np.array([r.wvtr_max for r in req]); r_light = np.array([r.light_min for r in req])
    r_seal = np.array([r.seal_min for r in req]); r_str = np.array([r.strength_min for r in req])
    nom = scaled(cands, np.arange(len(cands)), np.ones(len(cands)))
    feas = ((nom["otr23"].to_numpy()[None, :] <= r_otr_max[:, None]) & (nom["otr23"].to_numpy()[None, :] >= r_otr_min[:, None])
            & (nom["wvtr38"].to_numpy()[None, :] <= r_wvtr_max[:, None]) & (cands["light"].to_numpy()[None, :] >= r_light[:, None])
            & (cands["seal"].to_numpy()[None, :] >= r_seal[:, None]) & (cands["strength"].to_numpy()[None, :] >= r_str[:, None]))
    idx = rng.integers(0, len(cands), n)
    for i in np.where((rng.random(n) < 0.4) & feas.any(1))[0]:
        idx[i] = rng.choice(np.where(feas[i])[0])
    gs = np.exp(rng.uniform(math.log(0.5), math.log(2.0), n))
    sc = scaled(cands, idx, gs)
    pk = cands.iloc[idx].reset_index(drop=True)
    if not snack:
        # 70 % of the produce rows get a perforation area DESIGNED to put the in-pack O2 near a random target (6-20 %), with noise.
        # Without this most random produce packs are either sealed-and-anaerobic or wide open, and almost every label is 0.
        x = 0.209 - rng.uniform(0.004, 0.15, n)      # hole-dominated packs have O2 + CO2 ~ 21 %, so low O2 means damaging CO2
        perf_cm2 = perforation_cm2_for_o2(sc["otr23"].to_numpy(), area, mass, temp, resp_at_t, x) * np.exp(rng.normal(0, 0.3, n))
        designed = rng.random(n) < 0.7
        perf = np.where(designed, perf_cm2 * 100.0 / mass, perf)
    engine_pass = ((sc["otr23"] <= r_otr_max) & (sc["otr23"] >= r_otr_min) & (sc["wvtr38"] <= r_wvtr_max)
                   & (pk["light"] >= r_light) & (pk["seal"] >= r_seal) & (pk["strength"] >= r_str)).to_numpy()
    out = pd.DataFrame(dict(
        storage=np.where(chilled, "Chilled", "Ambient"), transport=np.where(harsh, "Harsh", "Normal"), harsh=harsh.astype(int),
        temp_c=temp, rh_pct=rh, target_days=target, resp_rate_mgco2_kg_h=resp_at_t,
        layers=pk["layers"], n_layers=pk["n_layers"], gauge_scale=gs, otr23=sc["otr23"], wvtr38=sc["wvtr38"],
        thickness_um=sc["thickness_um"], light_barrier=pk["light"], seal=pk["seal"], strength=pk["strength"],
        cost_index=pk["cost"], recyclable=pk["recyclable"].astype(int), breathable=pk["breathable"].astype(int),
        mass_kg=mass, area_m2_per_kg=area, perf_mm2_per_kg=perf, flush_o2_pct=flush, rule_engine_pass=engine_pass.astype(int)))
    return out


def generate(commodities: pd.DataFrame, n_rows: int, rng: np.random.Generator, cands: pd.DataFrame, split: str,
             n_trials: int, seed: int) -> pd.DataFrame:
    frames = []
    for _, c in commodities.iterrows():
        rows = _sample_rows(c, n_rows, rng, cands)
        for col in COMMODITY_COLS:
            rows[col] = c[col]
        rows["chilling_sensitive"] = float(c["chilling_sensitive"])
        rows["ph"] = c["ph"]
        rows["category"] = c["category"]
        rows["commodity"] = c["commodity"]
        rows["commodity_id"] = c["commodity_id"]
        rows["parent_id"] = c.get("parent_id", c["commodity_id"])
        rows["source"] = c.get("source", "real")
        rows["split"] = split
        frames.append(rows)
    df = pd.concat(frames, ignore_index=True)
    res = simulate(df, n_trials=n_trials, seed=seed)
    df = pd.concat([df, res.reset_index(drop=True)], axis=1)
    df["fail_rate"] = 1.0 - df["success_rate"]
    df["n_trials"] = n_trials
    return df


def assign_folds(parents: pd.DataFrame, rng: np.random.Generator) -> dict[str, int]:
    folds: dict[str, int] = {}
    for _, grp in parents.groupby("category"):
        ids = grp["commodity_id"].tolist()
        rng.shuffle(ids)
        for k, cid in enumerate(ids):
            folds[cid] = k % N_FOLDS
    return folds


# --------------------------------------------------------------------------- anchors

def supplied_anchor(master: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The 50 supplied shelf-life rows re-expressed as simulator inputs. Used only to compare simulated and observed life."""
    sl = pd.read_csv(SUPPLIED_SHELF_LIFE)
    cm = master.set_index("commodity")
    rows = []
    for _, r in sl.iterrows():
        name = r["Food_Type"]
        c = cm.loc[name]
        d = {k: c[k] for k in COMMODITY_COLS}
        d.update(commodity=name, category=c["category"], chilling_sensitive=float(c["chilling_sensitive"]),
                 temp_c=r["Storage_Temperature_C"], rh_pct=r["Relative_Humidity_Pct"], harsh=0, target_days=r["Shelf_Life_Days"],
                 otr23=r["OTR_cc_m2_day"], wvtr38=r["WVTR_g_m2_day"], light_barrier=0.1, seal=4, strength=3,
                 mass_kg=0.5, area_m2_per_kg=0.6, perf_mm2_per_kg=0.0, flush_o2_pct=20.9,
                 packaging=r["Packaging_Material_Type"], observed_shelf_life_days=r["Shelf_Life_Days"])
        rows.append(d)
    df = pd.DataFrame(rows)
    return pd.concat([df, simulate(df, n_trials=512, seed=seed).reset_index(drop=True)], axis=1)


def cookie_anchor(master: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Three-film cookie study (roadmap section 6): PP and PVdC/PP films compromised texture after 6 months at 22 C / 50 % RH,
    metallised PP extended acceptability. Cookie composition and pack geometry are NOT in the source: Cream Biscuits and a
    typical 100 g pack are assumed, so this is a direction check, not a calibration."""
    c = master.set_index("commodity").loc["Cream Biscuits"]
    films = [("PP film", 1307, 5.0, 0.1, False), ("acryl-coated PVdC/PP", 21.8, 4.2, 0.1, False), ("metallised PP", 31.2, 0.4, 0.95, True)]
    rows = []
    for name, o, w, lb, acceptable in films:
        d = {k: c[k] for k in COMMODITY_COLS}
        d.update(commodity="Cream Biscuits", category="dry_snack", chilling_sensitive=0.0, temp_c=22, rh_pct=50, harsh=0, target_days=182,
                 otr23=o, wvtr38=w, light_barrier=lb, seal=4, strength=3, mass_kg=0.1, area_m2_per_kg=1.5, perf_mm2_per_kg=0.0,
                 flush_o2_pct=20.9, film=name, observed_acceptable=acceptable)
        rows.append(d)
    df = pd.DataFrame(rows)
    return pd.concat([df, simulate(df, n_trials=512, seed=seed).reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------- main

def build(out: Path, seed: int = 0, n_real_train: int = 500, n_virtual_per_parent: int = 12, n_rows_virtual: int = 150,
          n_rows_test: int = 1000, n_trials_train: int = 128, n_trials_test: int = 256,
          test_commodities: list[str] | None = None) -> dict:
    """`test_commodities=[]` puts every commodity in train (spec 7.3: the deliverable model is refit on all of them after the
    hyperparameters are frozen); the test file is then written empty."""
    test_names = TEST_COMMODITIES if test_commodities is None else list(test_commodities)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    master = load_commodities()
    master["source"] = "real"
    master["parent_id"] = master["commodity_id"]
    cands = candidate_table(load_materials())
    test_c = master[master["commodity"].isin(test_names)]
    train_c = master[~master["commodity"].isin(test_names)]
    assert len(test_c) == len(test_names), "a TEST_COMMODITIES name is missing from the master table"
    folds = assign_folds(train_c, rng)

    virt = []
    for _, p in train_c.iterrows():
        for k in range(n_virtual_per_parent):
            v = make_virtual(p, k, rng)
            v["parent_id"] = p["commodity_id"]
            virt.append(v)
    virt_df = pd.DataFrame(virt)

    tr_real = generate(train_c, n_real_train, rng, cands, "train", n_trials_train, seed + 1)
    tr_virt = generate(virt_df, n_rows_virtual, rng, cands, "train", n_trials_train, seed + 2)
    train = pd.concat([tr_real, tr_virt], ignore_index=True)
    train["fold"] = train["parent_id"].map(folds)
    if len(test_c):
        test = generate(test_c, n_rows_test, rng, cands, "test", n_trials_test, seed + 3)
    else:
        test = train.iloc[:0].drop(columns=["fold"])
    test["fold"] = -1
    train.insert(0, "row_id", [f"tr{i:06d}" for i in range(len(train))])
    test.insert(0, "row_id", [f"te{i:06d}" for i in range(len(test))])

    # leakage guards
    assert set(train["parent_id"]).isdisjoint(set(test["parent_id"])), "commodity leaked across the split"
    assert set(train["commodity"].str.split("~").str[0]).isdisjoint(set(test["commodity"])), "virtual copy of a test commodity"

    sup = supplied_anchor(master, seed + 4)
    cookie = cookie_anchor(master, seed + 5)
    for name, df in (("train", train), ("test", test), ("anchor_supplied", sup), ("anchor_cookie", cookie)):
        df.to_csv(out / f"{name}.csv.gz", index=False, compression="gzip")
    master.drop(columns=["parent_id"]).to_csv(out / "commodities_master.csv", index=False)
    summary = dict(seed=seed, n_train=len(train), n_test=len(test), n_train_commodities_real=len(train_c), n_train_commodities_virtual=len(virt_df),
                   n_test_commodities=len(test_c), test_commodities=test_names, n_trials_train=n_trials_train, n_trials_test=n_trials_test,
                   n_candidate_structures=len(cands), folds=N_FOLDS,
                   # which commodity table produced the labels (composition from USDA FDC since 2026-10-26)
                   commodities_master_sha256=hashlib.sha256(MASTER_CSV.read_bytes()).hexdigest()[:16])
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small build for smoke tests")
    ap.add_argument("--no-test", action="store_true", help="every commodity in train, for the final model (spec 7.3)")
    a = ap.parse_args()
    kw = dict(n_real_train=40, n_virtual_per_parent=2, n_rows_virtual=20, n_rows_test=60) if a.quick else {}
    if a.no_test:
        kw["test_commodities"] = []
    print(json.dumps(build(Path(a.out), a.seed, **kw), indent=2))


if __name__ == "__main__":
    main()

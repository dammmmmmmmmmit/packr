"""Phase 5: gradient-boosted model of the simulator (spec section 7).

    python -m packr.ml.train --data data/datasets --out reports --models models          # tune, fit on train, score test once
    python -m packr.sim.dataset --out data/datasets_all --seed 0 --no-test               # regenerate with every commodity in train
    python -m packr.ml.train --final --data data/datasets_all --params reports/ml_metrics.json --models models

Protocol (spec 7.3): hyperparameters are chosen by grouped cross-validation on the train `fold` column only (scored on the real
rows of the held-out fold, like the sufficiency study). The chosen model is refit on all of train and scored on `test.csv.gz`
once. The acceptance criteria below are fixed before the test file is read and are not to be changed after.

What this model is: a fast surrogate of the SIMULATOR. Its accuracy says how well it reproduces the simulator, not how well
packaging works in reality (spec ground rule 3).
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..commodities import load_commodities
from ..profile import nearest
from .features import FEATURES, LATENT, LIFE_FLOOR_DAYS, featurise, featurise_with_profile

CRITERIA = dict(
    mae_success_max=0.10,            # held-out MAE of success rate
    auc_min=0.90,                    # AUC for "success rate >= 0.5"
    shelf_life_medae_log_max=0.30,   # median |ln(pred / simulated)| of shelf_life_p50
    calibration_bin_error_max=0.10,  # |mean predicted - mean simulated| in every one of 10 equal-width bins ...
    calibration_bin_min_rows=30,     # ... that holds at least this many test rows (emptier bins are reported, not judged)
)
BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
GRID = dict(num_leaves=[31, 63, 127], min_child_samples=[40, 200], trees=[(400, 0.05), (800, 0.03)])
QUICK_GRID = dict(num_leaves=[31], min_child_samples=[40], trees=[(150, 0.1)])
LIFE_HEADS = ("shelf_life_p10", "shelf_life_p50", "shelf_life_p90")
# The simulator's success rate cannot rise when only the target shelf life rises; the model is constrained to agree.
SUCCESS_MONOTONE = {"log_target_days": -1}


def grid_params(grid: dict) -> list[dict]:
    out = []
    for nl, mcs, (n, lr) in itertools.product(grid["num_leaves"], grid["min_child_samples"], grid["trees"]):
        out.append(dict(num_leaves=nl, min_child_samples=mcs, n_estimators=n, learning_rate=lr))
    return out


def _monotone_vector(cols: list[str], spec: dict) -> list[int]:
    return [spec.get(c, 0) for c in cols]


def fit_success(x: pd.DataFrame, y: pd.Series, params: dict, seed: int = 0) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="cross_entropy", monotone_constraints=_monotone_vector(list(x.columns), SUCCESS_MONOTONE),
                          random_state=seed, **BASE, **params)
    return m.fit(x, y)


def fit_life(x: pd.DataFrame, life: pd.Series, params: dict, seed: int = 0) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="regression", random_state=seed, **BASE, **params)
    return m.fit(x, np.log(life.clip(lower=LIFE_FLOOR_DAYS)))


def log_life(s: pd.Series) -> np.ndarray:
    return np.log(s.clip(lower=LIFE_FLOOR_DAYS)).to_numpy()


# --------------------------------------------------------------------------- tuning

def tune(train: pd.DataFrame, grid: dict, featuriser=featurise_with_profile) -> tuple[dict, dict, list[dict]]:
    """Grouped CV over `fold`. Returns (best success params, best life params, per-config results)."""
    x_all = featuriser(train)
    results = []
    for params in grid_params(grid):
        maes, aucs, lifes = [], [], []
        for f in sorted(train["fold"].unique()):
            tr = (train["fold"] != f).to_numpy()
            va = ((train["fold"] == f) & (train["source"] == "real")).to_numpy()
            ms = fit_success(x_all[tr], train.loc[tr, "success_rate"], params)
            ml = fit_life(x_all[tr], train.loc[tr, "shelf_life_p50"], params)
            p = np.clip(ms.predict(x_all[va]), 0, 1)
            y = train.loc[va, "success_rate"].to_numpy()
            maes.append(mean_absolute_error(y, p))
            aucs.append(roc_auc_score(y >= 0.5, p) if 0 < (y >= 0.5).mean() < 1 else np.nan)
            lifes.append(float(np.median(np.abs(ml.predict(x_all[va]) - log_life(train.loc[va, "shelf_life_p50"])))))
        results.append(dict(params=params, cv_mae=float(np.mean(maes)), cv_mae_sd=float(np.std(maes)), cv_mae_folds=[round(v, 4) for v in maes],
                            cv_auc=float(np.nanmean(aucs)), cv_life_medae_log=float(np.mean(lifes))))
        print(f"  {params}  cv MAE {results[-1]['cv_mae']:.4f} (sd {results[-1]['cv_mae_sd']:.4f})  life {results[-1]['cv_life_medae_log']:.3f}")
    best_s = min(results, key=lambda r: r["cv_mae"])["params"]
    best_l = min(results, key=lambda r: r["cv_life_medae_log"])["params"]
    return best_s, best_l, results


# --------------------------------------------------------------------------- model bundle

def fit_bundle(train: pd.DataFrame, success_params: dict, life_params: dict) -> dict:
    x = featurise_with_profile(train)
    heads = {"success_rate": fit_success(x, train["success_rate"], success_params)}
    for h in LIFE_HEADS:
        heads[h] = fit_life(x, train[h], life_params)
    return dict(heads=heads, columns=list(x.columns), features=FEATURES, latent=LATENT, success_params=success_params,
                life_params=life_params, trained_on=sorted(train.loc[train["source"] == "real", "commodity"].unique()),
                n_rows=len(train), created=date.today().isoformat(), lightgbm=lgb.__version__, python=platform.python_version(),
                note="surrogate of the uncalibrated simulator; simulation-based estimate, not validated against measured spoilage data")


# --------------------------------------------------------------------------- evaluation

def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append(dict(bin=f"{edges[b]:.1f}-{edges[b + 1]:.1f}", n=int(m.sum()),
                        mean_pred=float(p[m].mean()) if m.any() else None, mean_sim=float(y[m].mean()) if m.any() else None,
                        abs_err=float(abs(p[m].mean() - y[m].mean())) if m.any() else None))
    return out


def scores(test: pd.DataFrame, p: np.ndarray, life_pred: np.ndarray | None = None) -> dict:
    y = test["success_rate"].to_numpy()
    out = dict(mae=float(mean_absolute_error(y, p)), auc=float(roc_auc_score(y >= 0.5, p)))
    if life_pred is not None:
        out["life_medae_log"] = float(np.median(np.abs(life_pred - log_life(test["shelf_life_p50"]))))
    return out


def estimated_profiles(test: pd.DataFrame, master: pd.DataFrame, train_names: list[str]) -> pd.DataFrame:
    """Replace each test commodity's LATENT profile with the mean of its 3 nearest TRAIN commodities (the new-food case).
    Neighbours are found from what a user could supply: composition and the respiration rate (a plan input)."""
    pool = master[master["commodity"].isin(train_names)]
    out = test.copy()
    for name, grp in test.groupby("commodity"):
        r = master.set_index("commodity").loc[name]
        nn = nearest(pool, r["category"], r["moisture_pct"], r["fat_pct"],
                     resp_ref=r["resp_ref_mgco2_kg_h"] if r["category"] == "fresh_produce" else None)
        for c in LATENT:
            out.loc[grp.index, c] = nn[c].astype(float).mean()
    return out


def monotonic_wvtr_check(bundle: dict, test: pd.DataFrame, factor: float = 2.0) -> dict:
    """Spec T5: raising WVTR must not raise the mean predicted snack success rate."""
    snack = test[test["category"] == "dry_snack"]
    hi = snack.copy()
    hi["wvtr38"] = hi["wvtr38"] * factor
    from .predict import predict
    p0, p1 = predict(bundle, snack)["success_rate"].mean(), predict(bundle, hi)["success_rate"].mean()
    return dict(factor=factor, mean_success_base=float(p0), mean_success_higher_wvtr=float(p1), passed=bool(p1 <= p0 + 1e-9))


def evaluate(train: pd.DataFrame, test: pd.DataFrame, grid: dict, models: Path) -> dict:
    print("tuning on grouped folds (train only) ...")
    sp, lp, cv = tune(train, grid)
    best_cv = next(r for r in cv if r["params"] == sp)
    bundle = fit_bundle(train, sp, lp)
    joblib.dump(bundle, models / "packr_gbt_eval.joblib")

    from .predict import predict
    pred = predict(bundle, test)
    p, life50 = pred["success_rate"].to_numpy(), np.log(pred["shelf_life_p50"].clip(lower=LIFE_FLOOR_DAYS)).to_numpy()
    main = scores(test, p, life50)
    for h in ("shelf_life_p10", "shelf_life_p90"):
        main[f"{h}_medae_log"] = float(np.median(np.abs(np.log(pred[h].clip(lower=LIFE_FLOOR_DAYS)) - log_life(test[h]))))
    cal = calibration(test["success_rate"].to_numpy(), p)

    per_c = (test.assign(err=np.abs(test["success_rate"] - p), pred=p)
             .groupby(["category", "commodity"]).agg(mae=("err", "mean"), mean_sim=("success_rate", "mean"), mean_pred=("pred", "mean"))
             .round(4).reset_index())
    per_cat = {cat: scores(test[test.category == cat], p[(test.category == cat).to_numpy()], life50[(test.category == cat).to_numpy()])
               for cat in sorted(test["category"].unique())}

    # baselines
    cat_mean = train.groupby("category")["success_rate"].mean()
    y = test["success_rate"].to_numpy()
    plan_only_s = fit_success(featurise(train), train["success_rate"], sp)
    plan_only_l = fit_life(featurise(train), train["shelf_life_p50"], lp)
    master = load_commodities()
    est = estimated_profiles(test, master, sorted(train.loc[train["source"] == "real", "commodity"].unique()))
    pe = predict(bundle, est)
    baselines = dict(
        global_mean=dict(mae=float(mean_absolute_error(y, np.full(len(y), train["success_rate"].mean())))),
        category_mean=dict(mae=float(mean_absolute_error(y, test["category"].map(cat_mean)))),
        rule_engine_pass=scores(test, test["rule_engine_pass"].astype(float).to_numpy()),
        plan_inputs_only=scores(test, np.clip(plan_only_s.predict(featurise(test)), 0, 1), plan_only_l.predict(featurise(test))),
        estimated_profile_new_food=scores(test, pe["success_rate"].to_numpy(), np.log(pe["shelf_life_p50"].clip(lower=LIFE_FLOOR_DAYS)).to_numpy()),
    )
    judged = [b for b in cal if b["n"] >= CRITERIA["calibration_bin_min_rows"]]
    worst_bin = max(b["abs_err"] for b in judged)
    acceptance = {
        "mae_success": dict(value=main["mae"], limit=CRITERIA["mae_success_max"], passed=main["mae"] <= CRITERIA["mae_success_max"]),
        "auc": dict(value=main["auc"], limit=CRITERIA["auc_min"], passed=main["auc"] >= CRITERIA["auc_min"]),
        "shelf_life_log_medae": dict(value=main["life_medae_log"], limit=CRITERIA["shelf_life_medae_log_max"],
                                     passed=main["life_medae_log"] <= CRITERIA["shelf_life_medae_log_max"]),
        "calibration_worst_bin": dict(value=worst_bin, limit=CRITERIA["calibration_bin_error_max"],
                                      passed=worst_bin <= CRITERIA["calibration_bin_error_max"]),
    }
    p_floor = np.clip(y * (1 - y), 0, None)
    return dict(
        criteria=CRITERIA, success_params=sp, life_params=lp, cv_grid=cv,
        cv_chosen=dict(mae=best_cv["cv_mae"], sd=best_cv["cv_mae_sd"], folds=best_cv["cv_mae_folds"], auc=best_cv["cv_auc"],
                       life_medae_log=best_cv["cv_life_medae_log"]),
        test=main, per_category=per_cat, per_commodity=per_c.to_dict("records"), calibration=cal, baselines=baselines,
        acceptance={k: v | {"passed": bool(v["passed"])} for k, v in acceptance.items()},
        monotonic_wvtr_snack=monotonic_wvtr_check(bundle, test),
        mc_noise_floor_mae=float(np.mean(0.8 * np.sqrt(p_floor / test["n_trials"]))),
        n_train=len(train), n_test=len(test), test_commodities=sorted(test["commodity"].unique()),
        lightgbm=lgb.__version__)


# --------------------------------------------------------------------------- model card

def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def model_card(res: dict, dataset_summary: dict | None) -> str:
    a, t, b = res["acceptance"], res["test"], res["baselines"]
    lines = [
        "# Packr model card: success-rate and shelf-life surrogate (phase 5)", "",
        f"Generated by `python -m packr.ml.train` on {date.today().isoformat()} (LightGBM {res['lightgbm']}). "
        "Numbers come from `reports/ml_metrics.json`.", "",
        "## What this model is", "",
        "A LightGBM surrogate of the Packr Monte Carlo simulator (`packr.sim.physics`). It predicts the simulated success rate "
        "(share of trials reaching the target shelf life with no hard failure) and the simulated shelf-life p10/p50/p90.",
        "**Every simulator parameter is an assumption and the simulator is uncalibrated** (spec section 0). The figures below measure how "
        "well the model reproduces the simulator. They are **not** evidence about real packaging performance; that evidence is still "
        "almost absent (spec section 9.2). Every success rate shown to users is a *simulation-based estimate*.", "",
        "## Inputs", "",
        f"* {len(FEATURES)} plan-input and package features: `{', '.join(FEATURES)}`.",
        f"* {len(LATENT)} commodity-profile features (spec 7.2): `{', '.join(LATENT)}`. NaN where a parameter does not apply "
        "(snacks have no O2 floor, produce no critical aw).",
        "* At prediction time the profile comes from the commodity table, user overrides, or the mean of the 3 nearest table commodities "
        "(`packr.profile`). The last case is flagged `profile_source = estimated`.", "",
        "## Heads", "",
        "* `success_rate`: `objective=cross_entropy` on the fractional label, monotone non-increasing in `log_target_days`. Fail rate = 1 - success.",
        "* `shelf_life_p10`, `p50`, `p90`: three L2 regressors on log days (labels clipped at 0.05 d). Predictions are sorted so p10 <= p50 <= p90.",
        "* Failure-mode breakdown is **not** predicted; the API takes it from a direct 1,000-trial simulator run (spec 7.4).", "",
        "## Data and protocol", "",
        f"* Train: {res['n_train']:,} rows (real + virtual commodities). Test: {res['n_test']:,} rows from {len(res['test_commodities'])} held-out real "
        f"commodities: {', '.join(res['test_commodities'])}.",
        "* Split by commodity; virtual commodities come only from train commodities; CV folds are by parent commodity (spec section 5).",
        f"* Hyperparameters chosen by grouped 5-fold CV on train (grid of {len(res['cv_grid'])}), scored on the real rows of each held-out fold. "
        f"Success head: `{res['success_params']}`. Shelf-life heads: `{res['life_params']}`.",
        "* The test file was scored once, after the parameters were frozen. Anchors were not used.", "",
        "## Acceptance (spec 7.3, fixed before scoring)", "",
        "| Check | Value | Limit | Pass |", "|---|---|---|---|",
        f"| MAE of success rate, held-out commodities | {_fmt(a['mae_success']['value'])} | <= {a['mae_success']['limit']} | {'yes' if a['mae_success']['passed'] else '**no**'} |",
        f"| AUC (success >= 0.5) | {_fmt(a['auc']['value'])} | >= {a['auc']['limit']} | {'yes' if a['auc']['passed'] else '**no**'} |",
        f"| Median abs log error, shelf life p50 | {_fmt(a['shelf_life_log_medae']['value'])} | <= {a['shelf_life_log_medae']['limit']} | {'yes' if a['shelf_life_log_medae']['passed'] else '**no**'} |",
        f"| Worst calibration bin (bins with >= {res['criteria']['calibration_bin_min_rows']} rows) | {_fmt(a['calibration_worst_bin']['value'])} | <= {a['calibration_worst_bin']['limit']} | {'yes' if a['calibration_worst_bin']['passed'] else '**no**'} |",
        "",
        f"Grouped CV on train with the chosen parameters: MAE {_fmt(res['cv_chosen']['mae'])} (sd {_fmt(res['cv_chosen']['sd'])}, folds {res['cv_chosen']['folds']}). "
        f"Monte Carlo label-noise floor on test: MAE about {_fmt(res['mc_noise_floor_mae'])}.", "",
        "## Baselines on the same test rows", "",
        "| Model | MAE | AUC | Shelf-life log error |", "|---|---|---|---|",
        f"| **This model, profile from the table** | {_fmt(t['mae'])} | {_fmt(t['auc'])} | {_fmt(t['life_medae_log'])} |",
        f"| This model, profile **estimated** from 3 nearest train commodities (new-food case) | {_fmt(b['estimated_profile_new_food']['mae'])} | {_fmt(b['estimated_profile_new_food']['auc'])} | {_fmt(b['estimated_profile_new_food']['life_medae_log'])} |",
        f"| Same GBT, plan inputs only (no profile; the section 6 configuration) | {_fmt(b['plan_inputs_only']['mae'])} | {_fmt(b['plan_inputs_only']['auc'])} | {_fmt(b['plan_inputs_only']['life_medae_log'])} |",
        f"| Rule-engine pass/fail | {_fmt(b['rule_engine_pass']['mae'])} | {_fmt(b['rule_engine_pass']['auc'])} | |",
        f"| Category mean | {_fmt(b['category_mean']['mae'])} | | |",
        f"| Global mean | {_fmt(b['global_mean']['mae'])} | | |",
        "",
        "With the profile taken from the table, the model sees the same commodity parameters as the section 6 *oracle*, so the first row "
        "is the oracle configuration. The second row is the honest figure for a food that is **not** in the commodity table.", "",
        "## Per category and per commodity (test)", "",
        "| Category | MAE | AUC | Shelf-life log error |", "|---|---|---|---|",
        *[f"| {c} | {_fmt(v['mae'])} | {_fmt(v['auc'])} | {_fmt(v['life_medae_log'])} |" for c, v in res["per_category"].items()],
        "", "| Commodity | Category | MAE | Mean simulated | Mean predicted |", "|---|---|---|---|---|",
        *[f"| {r['commodity']} | {r['category']} | {_fmt(r['mae'])} | {_fmt(r['mean_sim'])} | {_fmt(r['mean_pred'])} |"
          for r in sorted(res["per_commodity"], key=lambda r: r["mae"])],
        "", "## Calibration (test, 10 equal-width bins of predicted success)", "",
        "| Bin | Rows | Mean predicted | Mean simulated | Abs error |", "|---|---|---|---|---|",
        *[f"| {c['bin']} | {c['n']} | {_fmt(c['mean_pred'])} | {_fmt(c['mean_sim'])} | {_fmt(c['abs_err'])} |" for c in res["calibration"]],
        "", "## Monotonicity check", "",
        f"Doubling WVTR on every snack test row moves the mean predicted snack success from {_fmt(res['monotonic_wvtr_snack']['mean_success_base'])} "
        f"to {_fmt(res['monotonic_wvtr_snack']['mean_success_higher_wvtr'])} ({'passes' if res['monotonic_wvtr_snack']['passed'] else '**fails**'}: it must not rise).", "",
        "## Limits", "",
        "* Inherits every simulator limit (spec 4.3): no frozen storage, no ethylene, no microbial model, steady-state produce gas balance, "
        "linear snack isotherm, placeholder material values.",
        "* Covers two categories only (fresh produce, dry snacks); category extrapolation fails (spec section 6).",
        "* The training distribution is a sampling design, not real requests (spec 11.8). Temperatures outside 0 to 38 C, targets above 400 days, "
        "or packs far from the sampled geometry are extrapolation.",
        "* The deliverable model is refit on all 48 commodities (`--final`); its test-set numbers are the ones above, from the held-out run.",
    ]
    if dataset_summary:
        lines += ["", f"Dataset: seed {dataset_summary.get('seed')}, {dataset_summary.get('n_train'):,} train / {dataset_summary.get('n_test'):,} test rows."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="packr.ml.train")
    ap.add_argument("--data", default="data/datasets")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--models", default="models")
    ap.add_argument("--quick", action="store_true", help="one-point grid, for smoke tests")
    ap.add_argument("--final", action="store_true", help="refit on train + test of --data with the frozen parameters in --params")
    ap.add_argument("--params", default="reports/ml_metrics.json")
    a = ap.parse_args(argv)
    data, out, models = Path(a.data), Path(a.out), Path(a.models)
    models.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv.gz")
    summary_path = data / "dataset_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    if a.final:
        frozen = json.loads(Path(a.params).read_text())
        test_path = data / "test.csv.gz"
        test = pd.read_csv(test_path) if test_path.exists() else train.iloc[:0]
        full = pd.concat([train, test], ignore_index=True) if len(test) else train      # an empty test file reads as object dtypes
        bundle = fit_bundle(full, frozen["success_params"], frozen["life_params"])
        bundle["dataset_summary"] = summary
        joblib.dump(bundle, models / "packr_gbt.joblib")
        print(f"wrote {models / 'packr_gbt.joblib'}: {len(full):,} rows, {len(bundle['trained_on'])} real commodities")
        return

    out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(data / "test.csv.gz")
    res = evaluate(train, test, QUICK_GRID if a.quick else GRID, models)
    res["dataset_summary"] = summary
    (out / "ml_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "model_card.md").write_text(model_card(res, summary), encoding="utf-8")
    print(json.dumps({"test": res["test"], "acceptance": res["acceptance"], "baselines": res["baselines"],
                      "cv_chosen": res["cv_chosen"]}, indent=2, default=float))


if __name__ == "__main__":
    main()

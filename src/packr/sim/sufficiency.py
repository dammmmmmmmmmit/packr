"""Is there enough data for the planned architecture (gradient-boosted trees, optional FT-Transformer)?

    python -m packr.sim.sufficiency --data data/datasets --out reports

"Enough" is judged on HELD-OUT COMMODITIES, because rows are cheap (the simulator makes millions) while distinct commodities are
scarce and are what a new request actually varies. Acceptance criteria are fixed here, before the numbers are seen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..ml.features import FEATURES, LATENT, featurise  # noqa: F401  (FEATURES and LATENT are part of this module's interface)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CRITERIA = dict(
    mae_success_max=0.10,           # mean absolute error of predicted success rate on held-out commodities
    improvement_vs_best_baseline=0.30,   # must beat the best simple baseline by this relative margin
    plateau_gain_max=0.05,          # going from 75 % to 100 % of the commodities may improve MAE by less than this (relative)
    auc_min=0.90,                   # AUC for "success rate >= 0.5" on held-out commodities
    shelf_life_medae_log_max=0.30,  # median |ln(pred / simulated)| of shelf life (0.30 ~ 35 % error)
)
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=4)


def fit(train: pd.DataFrame, cols: list[str], target: str, extra: pd.DataFrame | None = None, objective: str = "cross_entropy"):
    x = featurise(train)
    if extra is not None:
        x = pd.concat([x, extra], axis=1)
    y = train[target] if target == "success_rate" else np.log(train[target].clip(lower=0.05))
    m = lgb.LGBMRegressor(objective=objective if target == "success_rate" else "regression", **PARAMS)
    m.fit(x, y)
    return m


def evaluate(train: pd.DataFrame, test: pd.DataFrame, latent: bool = False) -> dict:
    ex_tr = train[LATENT].fillna(-1) if latent else None
    ex_te = test[LATENT].fillna(-1) if latent else None
    ms = fit(train, FEATURES, "success_rate", ex_tr)
    ml = fit(train, FEATURES, "shelf_life_p50", ex_tr)
    xt = featurise(test)
    if latent:
        xt = pd.concat([xt, ex_te], axis=1)
    ps, pl = np.clip(ms.predict(xt), 0, 1), ml.predict(xt)
    return dict(mae=float(mean_absolute_error(test["success_rate"], ps)),
                auc=float(roc_auc_score(test["success_rate"] >= 0.5, ps)),
                life_medae_log=float(np.median(np.abs(pl - np.log(test["shelf_life_p50"].clip(lower=0.05))))),
                pred=ps, pred_life=pl)


def baselines(train, test) -> dict:
    cat_mean = train.groupby("category")["success_rate"].mean()
    return dict(
        global_mean=float(mean_absolute_error(test["success_rate"], np.full(len(test), train["success_rate"].mean()))),
        category_mean=float(mean_absolute_error(test["success_rate"], test["category"].map(cat_mean))),
        rule_engine=float(mean_absolute_error(test["success_rate"], test["rule_engine_pass"].astype(float))),
        rule_engine_auc=float(roc_auc_score(test["success_rate"] >= 0.5, test["rule_engine_pass"])))


def subset_commodities(train, k, rng, virtual=True):
    parents = train.drop_duplicates("parent_id")[["parent_id", "category"]]
    snack = parents[parents.category == "dry_snack"]["parent_id"].to_numpy()
    prod = parents[parents.category == "fresh_produce"]["parent_id"].to_numpy()
    ks = max(2, round(k * len(snack) / len(parents)))
    keep = set(rng.choice(snack, min(ks, len(snack)), replace=False)) | set(rng.choice(prod, min(k - ks, len(prod)), replace=False))
    m = train["parent_id"].isin(keep)
    if not virtual:
        m &= train["source"] == "real"
    return train[m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(data / "train.csv.gz"), pd.read_csv(data / "test.csv.gz")
    res: dict = {"criteria": CRITERIA}

    # A. main fit + baselines + noise floor
    main_r = evaluate(train, test)
    base = baselines(train, test)
    p = test["success_rate"].to_numpy()
    floor = float(np.mean(0.8 * np.sqrt(np.clip(p * (1 - p), 0, None) / test["n_trials"])))
    res["main"] = {k: v for k, v in main_r.items() if not k.startswith("pred")} | {"baselines": base, "mc_noise_floor_mae": floor}
    by_c = (test.assign(err=np.abs(test["success_rate"] - main_r["pred"]))
            .groupby(["category", "commodity"])["err"].mean().round(3).reset_index())
    res["per_commodity_mae"] = by_c.to_dict("records")
    res["per_category_mae"] = by_c.groupby("category")["err"].mean().round(3).to_dict()

    # B. learning curve vs number of training commodities (real + virtual copies, and real rows only)
    n_parents = train["parent_id"].nunique()
    ks = sorted({6, 12, 18, 24, 30, n_parents})
    lc = []
    for k in ks:
        for rep in range(3 if k < n_parents else 1):
            rng = np.random.default_rng(100 * k + rep)
            for virtual in (True, False):
                sub = subset_commodities(train, k, rng, virtual)
                r = evaluate(sub, test)
                lc.append(dict(k=k, rep=rep, virtual=virtual, rows=len(sub), mae=r["mae"], auc=r["auc"], life=r["life_medae_log"]))
    lc = pd.DataFrame(lc)
    res["learning_curve_commodities"] = lc.groupby(["virtual", "k"])[["rows", "mae", "auc", "life"]].mean().round(4).reset_index().to_dict("records")

    # C. learning curve vs rows at all commodities
    rows_lc = []
    for frac in (0.01, 0.03, 0.1, 0.3, 1.0):
        sub = train.sample(frac=frac, random_state=0)
        r = evaluate(sub, test)
        rows_lc.append(dict(frac=frac, rows=len(sub), mae=r["mae"], auc=r["auc"]))
    res["learning_curve_rows"] = rows_lc

    # D. oracle: model also sees the latent commodity parameters
    orc = evaluate(train, test, latent=True)
    res["oracle_with_latent_params"] = {k: v for k, v in orc.items() if not k.startswith("pred")}

    # E. category extrapolation
    cross = {}
    for src, dst in (("fresh_produce", "dry_snack"), ("dry_snack", "fresh_produce")):
        tr_c, te_c = train[train.category == src], test[test.category == dst]
        m = fit(tr_c, FEATURES, "success_rate")
        cross[f"train_{src}_test_{dst}_mae"] = float(mean_absolute_error(te_c["success_rate"], np.clip(m.predict(featurise(te_c)), 0, 1)))
    res["category_extrapolation"] = cross

    # F. grouped CV on train folds
    cv = []
    for f in sorted(train["fold"].unique()):
        r = evaluate(train[train.fold != f], train[(train.fold == f) & (train.source == "real")])
        cv.append(r["mae"])
    res["grouped_cv_mae"] = dict(folds=[round(v, 4) for v in cv], mean=float(np.mean(cv)), sd=float(np.std(cv)))

    # G. coverage of test commodities in property space vs train commodities
    cm = pd.concat([train, test]).drop_duplicates("commodity_id")
    cm = cm[cm["source"] == "real"]
    cols = ["moisture_pct", "fat_pct", "ph", "resp_ref_mgco2_kg_h", "base_life_days", "q10_decay"]
    z = cm[cols].copy(); z["resp_ref_mgco2_kg_h"] = np.log1p(z["resp_ref_mgco2_kg_h"]); z["base_life_days"] = np.log(z["base_life_days"])
    z = (z - z.mean()) / z.std().replace(0, 1)
    is_test = cm["split"].eq("test").to_numpy(); cats = cm["category"].to_numpy()
    nn = []
    for i in np.where(is_test)[0]:
        d = np.linalg.norm(z.fillna(0).to_numpy()[~is_test & (cats == cats[i])] - z.fillna(0).to_numpy()[i], axis=1)
        nn.append(d.min())
    loo = []
    for i in np.where(~is_test)[0]:
        pool = ~is_test & (cats == cats[i]); pool[i] = False
        loo.append(np.linalg.norm(z.fillna(0).to_numpy()[pool] - z.fillna(0).to_numpy()[i], axis=1).min())
    res["coverage"] = dict(real_commodities=cm.groupby("category").size().to_dict(), test_nn_distance_median=float(np.median(nn)),
                           train_leave_one_out_nn_distance_median=float(np.median(loo)))

    # H. literature anchors
    sup = pd.read_csv(data / "anchor_supplied.csv.gz")
    sup["ratio"] = sup["shelf_life_p50"] / sup["observed_shelf_life_days"]
    nm = sup[sup["packaging"] != "MAP High Barrier"]
    mp = sup[sup["packaging"] == "MAP High Barrier"]
    res["anchor_supplied"] = dict(
        non_map_rows=len(nm), non_map_median_ratio=float(nm.ratio.median()), non_map_spearman=float(nm[["observed_shelf_life_days", "shelf_life_p50"]].corr("spearman").iloc[0, 1]),
        map_rows=len(mp), map_median_ratio=float(mp.ratio.median()),
        note="base_life_days of foods that exist only in the supplied CSV was taken from its own non-MAP rows, so the non-MAP agreement is NOT independent validation")
    ck = pd.read_csv(data / "anchor_cookie.csv.gz")
    res["anchor_cookie"] = ck[["film", "observed_acceptable", "success_rate", "shelf_life_p50"]].round(3).to_dict("records")
    from .physics import simulate
    from ..commodities import load_commodities
    cmast = load_commodities().set_index("commodity").loc["Cream Biscuits"]
    sweep = []
    for aw in np.arange(0.20, 0.61, 0.05):
        d = ck.copy(); d["crit_aw"] = aw
        r = simulate(d, n_trials=512, seed=7)["success_rate"].round(2).tolist()
        sweep.append(dict(crit_aw=round(float(aw), 2), success_rates=r))
    res["anchor_cookie_crit_aw_sweep"] = sweep

    # verdict
    best_base = min(base["category_mean"], base["global_mean"], base["rule_engine"])
    v = main_r
    k_hi = lc[(lc.virtual) & (lc.k == n_parents)]["mae"].mean()
    k_75 = lc[(lc.virtual) & (lc.k == 30)]["mae"].mean()
    gain = (k_75 - k_hi) / k_75
    checks = {
        "mae_success": (v["mae"], v["mae"] <= CRITERIA["mae_success_max"]),
        "beats_baseline": (1 - v["mae"] / best_base, 1 - v["mae"] / best_base >= CRITERIA["improvement_vs_best_baseline"]),
        "plateau_in_commodities": (gain, gain <= CRITERIA["plateau_gain_max"]),
        "auc": (v["auc"], v["auc"] >= CRITERIA["auc_min"]),
        "shelf_life_log_medae": (v["life_medae_log"], v["life_medae_log"] <= CRITERIA["shelf_life_medae_log_max"]),
    }
    res["verdict"] = {k: dict(value=round(float(a_), 4), passed=bool(b)) for k, (a_, b) in checks.items()}
    (out / "sufficiency.json").write_text(json.dumps(res, indent=2, default=float))

    # figures
    ink, mute = "#1f2937", "#6b7280"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for virtual, col, lab in ((True, "#2563eb", "real + virtual copies"), (False, "#d97706", "real rows only")):
        g = lc[lc.virtual == virtual].groupby("k")["mae"].agg(["mean", "min", "max"])
        ax[0].plot(g.index, g["mean"], color=col, marker="o", lw=2)
        ax[0].fill_between(g.index, g["min"], g["max"], color=col, alpha=0.12)
        ax[0].annotate(lab, (g.index[-1], g["mean"].iloc[-1]), textcoords="offset points", xytext=(-4, 8 if virtual else -14), ha="right", color=ink, fontsize=9)
    ax[0].axhline(CRITERIA["mae_success_max"], color=mute, ls=":", lw=1); ax[0].text(ks[0], CRITERIA["mae_success_max"] + 0.004, "target MAE 0.10", color=mute, fontsize=8)
    ax[0].axhline(floor, color=mute, ls=":", lw=1); ax[0].text(ks[0], floor + 0.004, "Monte Carlo noise floor", color=mute, fontsize=8)
    ax[0].set_xlabel("training commodities"); ax[0].set_ylabel("held-out MAE of success rate"); ax[0].set_title("Error vs number of commodities", loc="left", fontsize=10, color=ink)
    ax[1].scatter(test["success_rate"], main_r["pred"], s=3, alpha=0.25, color="#2563eb", linewidths=0)
    ax[1].plot([0, 1], [0, 1], color=mute, lw=1)
    ax[1].set_xlabel("simulated success rate"); ax[1].set_ylabel("predicted"); ax[1].set_title("Held-out commodities", loc="left", fontsize=10, color=ink)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False); a_.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(out / "sufficiency.png", dpi=150)
    print(json.dumps({"main": res["main"], "verdict": res["verdict"], "oracle": res["oracle_with_latent_params"],
                      "category_extrapolation": cross, "cv": res["grouped_cv_mae"], "coverage": res["coverage"]}, indent=2, default=float))


if __name__ == "__main__":
    main()

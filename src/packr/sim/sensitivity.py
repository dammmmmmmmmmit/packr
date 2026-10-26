"""Global sensitivity analysis of the simulator (spec T3, part A). No behaviour change: it only perturbs inputs.

    python -m packr.sim.sensitivity --data data/datasets --out reports

Method: Morris elementary effects (SALib), run separately for dry_snack and fresh_produce on a sample of REAL train rows (equal
rows per commodity). Each factor is either a commodity simulator parameter (perturbed on every row at once) or a Constants field.
Outputs: mean success rate and mean ln(shelf_life_p50) over the sample. Every evaluation uses the same random seed (common random
numbers), so differences come from the factors, not from Monte Carlo noise.

RANGES ARE A CHOICE and drive the ranking: every assumed scale parameter is varied over x0.5 to x2 (log scale); bounded ones get the
additive ranges below. Composition (moisture, fat) is sourced from USDA FDC since 2026-10-26 and is varied only +-20 % for context.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import CONSTANTS, Constants, simulate

# (name, kind, low, high). kind: "scale" multiplies by 2**u (low/high are log2 bounds); "add" adds u; "rel" multiplies by (1 + u).
SNACK_PARAMS = [("crit_aw", "add", -0.10, 0.10), ("iso_b", "scale", -1, 1), ("ox_ref_days", "scale", -1, 1),
                ("ea_ox_kj", "scale", -1, 1), ("light_sens", "scale", -1, 1), ("fragility", "scale", -1, 1),
                ("moisture_pct", "rel", -0.2, 0.2), ("fat_pct", "rel", -0.2, 0.2)]
PRODUCE_PARAMS = [("resp_ref_mgco2_kg_h", "scale", -1, 1), ("resp_ref_temp_c", "add", -5, 5), ("q10_resp", "add", -0.5, 0.5),
                  ("ideal_temp_c", "add", -3, 3), ("base_life_days", "scale", -1, 1), ("fragility", "scale", -1, 1),
                  ("o2_min_pct", "scale", -1, 1), ("co2_max_pct", "scale", -1, 1), ("crit_weight_loss_pct", "scale", -1, 1),
                  ("ma_gain", "scale", -1, 1), ("transp_g_kg_d", "scale", -1, 1), ("q10_decay", "add", -0.5, 0.5),
                  ("moisture_pct", "rel", -0.2, 0.2)]
SOURCED = {"moisture_pct", "fat_pct"}
# Constants: all x0.5..x2 except o2_air (a physical constant, excluded). leak_prob_by_seal is scaled as a whole.
CONSTANT_FIELDS = [f.name for f in dataclasses.fields(Constants) if f.name != "o2_air" and not f.metadata.get("structural")]
BOUNDS = {"crit_aw": (0.05, 0.95), "q10_resp": (1.2, 4.0), "q10_decay": (1.2, 4.0)}


def factors(category: str) -> list[tuple[str, str, float, float, str]]:
    params = SNACK_PARAMS if category == "dry_snack" else PRODUCE_PARAMS
    out = [(n, k, lo, hi, "commodity") for n, k, lo, hi in params]
    out += [(n, "scale", -1, 1, "constant") for n in CONSTANT_FIELDS]
    return out


def representative_rows(train: pd.DataFrame, category: str, n: int = 300, seed: int = 0) -> pd.DataFrame:
    real = train[(train["category"] == category) & (train["source"] == "real")]
    per = max(1, n // real["commodity"].nunique())
    return real.groupby("commodity", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed)).reset_index(drop=True)


def _apply(rows: pd.DataFrame, fac: list, x: np.ndarray) -> tuple[pd.DataFrame, Constants]:
    d = rows.copy()
    changes = {}
    for (name, kind, _, _, where), u in zip(fac, x):
        if where == "commodity":
            v = d[name].astype(float)
            v = v * 2.0 ** u if kind == "scale" else (v + u if kind == "add" else v * (1 + u))
            if name in BOUNDS:
                v = v.clip(*BOUNDS[name])
            if name == "moisture_pct":
                v = v.clip(0.1, 99.0)
            d[name] = v
        else:
            base = getattr(CONSTANTS, name)
            changes[name] = tuple(min(p * 2.0 ** u, 1.0) for p in base) if isinstance(base, tuple) else base * 2.0 ** u
    return d, dataclasses.replace(CONSTANTS, **changes)


def evaluate(rows: pd.DataFrame, fac: list, X: np.ndarray, n_trials: int = 64, seed: int = 11) -> np.ndarray:
    out = np.empty((len(X), 2))
    for i, x in enumerate(X):
        d, c = _apply(rows, fac, x)
        r = simulate(d, n_trials=n_trials, seed=seed, c=c)                  # same seed every time: common random numbers
        out[i] = (r["success_rate"].astype(float).mean(), np.log(r["shelf_life_p50"].astype(float).clip(lower=0.05)).mean())
    return out


def morris(rows: pd.DataFrame, category: str, trajectories: int = 40, levels: int = 8, seed: int = 0, n_trials: int = 64) -> pd.DataFrame:
    from SALib.analyze import morris as morris_analyze
    from SALib.sample import morris as morris_sample
    fac = factors(category)
    problem = {"num_vars": len(fac), "names": [f[0] if f[4] == "commodity" else f"Constants.{f[0]}" for f in fac],
               "bounds": [[f[2], f[3]] for f in fac]}
    X = morris_sample.sample(problem, trajectories, num_levels=levels, seed=seed)
    Y = evaluate(rows, fac, X, n_trials=n_trials)
    res = []
    for j, label in enumerate(("success_rate", "log_shelf_life_p50")):
        a = morris_analyze.analyze(problem, X, Y[:, j], num_levels=levels, seed=seed)
        res.append(pd.DataFrame({"factor": problem["names"], f"mu_star_{label}": a["mu_star"], f"sigma_{label}": a["sigma"],
                                 f"mu_star_conf_{label}": a["mu_star_conf"]}))
    df = res[0].merge(res[1], on="factor")
    meta = {(f[0] if f[4] == "commodity" else f"Constants.{f[0]}"): f for f in fac}
    df["kind"] = df["factor"].map(lambda n: meta[n][4])
    df["range"] = df["factor"].map(lambda n: _range_text(meta[n]))
    df["sourced"] = df["factor"].isin(SOURCED)
    df["rank_success"] = df["mu_star_success_rate"].rank(ascending=False, method="min").astype(int)
    df["rank_life"] = df["mu_star_log_shelf_life_p50"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["rank_success", "rank_life"]).reset_index(drop=True)


def _range_text(f) -> str:
    _, kind, lo, hi, _ = f
    return f"x{2 ** lo:g} to x{2 ** hi:g}" if kind == "scale" else (f"{lo:+g} to {hi:+g}" if kind == "add" else f"{lo:+.0%} to {hi:+.0%}")


def report(results: dict[str, pd.DataFrame], meta: dict) -> str:
    lines = ["# Simulator sensitivity (Morris elementary effects)", "",
             f"Generated by `python -m packr.sim.sensitivity`. Sample: {meta['rows']} real train rows per category (equal per commodity), "
             f"{meta['trials']} Monte Carlo trials per row, {meta['trajectories']} Morris trajectories, {meta['levels']} levels, common random numbers.", "",
             "`mu*` = mean absolute elementary effect when the factor moves across its whole range: the change in mean success rate "
             "(0 to 1) or in mean ln(shelf-life p50). `sigma` large relative to `mu*` means non-linear or interacting. "
             "**Ranges drive the ranking**: assumed scale parameters x0.5 to x2, bounded ones additive (column `range`). "
             "Composition (moisture, fat) comes from USDA FDC and is varied only +-20 %, so it is shown for context and marked *sourced*.",
             "", "This ranking is the order for literature look-ups (spec section 11). It says nothing about which values are right.",
             "", "Coupled factors: `Constants.d_o2_air` / `Constants.film_hole_cm` and `Constants.d_wv_air` / `Constants.seal_channel_cm` enter "
             "the physics only as diffusion-coefficient-over-path-length ratios (hole conductance), so each pair is one quantity to look up.", ""]
    for cat, df in results.items():
        lines += [f"## {cat}", "", "| Rank | Factor | Range | mu* success | sigma | mu* ln life | Rank (life) |", "|---|---|---|---|---|---|---|"]
        for r in df.itertuples():
            name = f"{r.factor} *(sourced)*" if r.sourced else r.factor
            lines.append(f"| {r.rank_success} | {name} | {r.range} | {r.mu_star_success_rate:.3f} | {r.sigma_success_rate:.3f} | "
                         f"{r.mu_star_log_shelf_life_p50:.3f} | {r.rank_life} |")
        inactive = df[(df["mu_star_success_rate"] < 1e-4) & (df["mu_star_log_shelf_life_p50"] < 1e-4)]["factor"].tolist()
        lines += ["", f"No effect in this category (mu* < 1e-4 on both outputs): {', '.join(inactive) or 'none'}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="packr.sim.sensitivity")
    ap.add_argument("--data", default="data/datasets")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--trajectories", type=int, default=40)
    ap.add_argument("--trials", type=int, default=64)
    a = ap.parse_args(argv)
    train = pd.read_csv(Path(a.data) / "train.csv.gz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cat in ("dry_snack", "fresh_produce"):
        rows = representative_rows(train, cat, a.rows)
        results[cat] = morris(rows, cat, trajectories=a.trajectories, n_trials=a.trials)
        print(cat, len(rows), "rows; top 8:", results[cat]["factor"].head(8).tolist())
    meta = dict(rows=a.rows, trials=a.trials, trajectories=a.trajectories, levels=8)
    (out / "sensitivity.md").write_text(report(results, meta), encoding="utf-8")
    (out / "sensitivity.json").write_text(json.dumps({k: v.to_dict("records") for k, v in results.items()} | {"meta": meta}, indent=2, default=float))


if __name__ == "__main__":
    main()

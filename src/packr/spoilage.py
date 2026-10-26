"""Fruit spoilage / environment sensor dataset (Mendeley czz68d9fwj).

Columns in the source file: fruit, temp, humid (%), light (fux), co2 (pmm), class (Good/Bad).
There is NO packaging variable in it, so it cannot supply packaging -> outcome labels. At most it
informs how temperature, humidity and CO2 relate to spoilage for the four fruits it covers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_PREFIXES = {"fruit": "fruit", "temp": "temp_c", "humid": "humidity_pct", "light": "light_lux",
             "co2": "co2_ppm", "class": "label"}
FEATURES = ["temp_c", "humidity_pct", "light_lux", "co2_ppm"]


def load_spoilage(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        for prefix, canonical in _PREFIXES.items():
            if key.startswith(prefix):
                rename[col] = canonical
    df = df.rename(columns=rename)
    missing = set(_PREFIXES.values()) - set(df.columns)
    if missing:
        raise ValueError(f"{path} does not look like the spoilage dataset: missing {sorted(missing)}; "
                         f"found {list(df.columns)}")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def summarise_by_fruit(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("fruit")
              .agg(avg_temp_c=("temp_c", "mean"), avg_humidity_pct=("humidity_pct", "mean"),
                   avg_co2_ppm=("co2_ppm", "mean"), avg_light_lux=("light_lux", "mean"),
                   n_records=("label", "count"), pct_good=("label", lambda s: (s == "good").mean() * 100))
              .reset_index().rename(columns={"fruit": "commodity"}))


def audit_spoilage(df: pd.DataFrame, folds: int = 5, seed: int = 0) -> str:
    """Describe the dataset and test whether its features predict the label at all."""
    lines = [f"SPOILAGE: {len(df)} rows, {df['fruit'].nunique()} fruits ({', '.join(sorted(df['fruit'].unique()))})",
             "no packaging column present: " + str(not any("pack" in c for c in df.columns)),
             "feature ranges: " + "; ".join(f"{c} {df[c].min():.3g}..{df[c].max():.3g}" for c in FEATURES),
             "share labelled good: " + ", ".join(f"{f}={p:.0f}%" for f, p in
                                                 df.groupby("fruit")["label"].apply(lambda s: (s == 'good').mean() * 100).items())]
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        return "\n".join(lines + ["cross-validated AUC skipped: scikit-learn is not installed"])
    y = (df["label"] == "good").astype(int)
    X = pd.concat([df[FEATURES], pd.get_dummies(df["fruit"], dtype=float)], axis=1)
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    auc = cross_val_score(model, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                          scoring="roc_auc").mean()
    lines.append(f"{folds}-fold cross-validated AUC of fruit + sensor features -> good/bad: {auc:.3f} "
                 "(0.5 = no better than chance; if close to 0.5 this dataset cannot train a spoilage model)")
    return "\n".join(lines)

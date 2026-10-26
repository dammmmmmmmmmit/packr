"""Load a trained model bundle and predict success rate and shelf life for prepared rows.

Rows need the columns `featurise` reads (plan inputs + package spec, see `packr.ml.features`) and the LATENT profile columns.
`packr.optimize.build_rows` produces such rows from a Scenario, a Profile and candidate packages.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..paths import MODELS_DIR
from .features import featurise_with_profile

DEFAULT_MODEL = "packr_gbt.joblib"          # refit on all commodities (`train --final`)
EVAL_MODEL = "packr_gbt_eval.joblib"        # fitted on train commodities only (`train`), used if the final model is absent


def default_model_path() -> Path | None:
    env = os.environ.get("PACKR_MODEL")
    if env:
        return Path(env)
    for name in (DEFAULT_MODEL, EVAL_MODEL):
        if (MODELS_DIR / name).exists():
            return MODELS_DIR / name
    return None


def load_model(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else default_model_path()
    if path is None or not path.exists():
        raise FileNotFoundError(f"no trained model found (looked for {path or MODELS_DIR / DEFAULT_MODEL}); run `python -m packr.ml.train`")
    bundle = joblib.load(path)
    bundle["path"] = str(path)
    return bundle


def predict(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    x = featurise_with_profile(rows)[bundle["columns"]]
    heads = bundle["heads"]
    p = np.clip(heads["success_rate"].predict(x), 0.0, 1.0)
    life = np.sort(np.column_stack([heads[h].predict(x) for h in ("shelf_life_p10", "shelf_life_p50", "shelf_life_p90")]), axis=1)
    life = np.exp(life)
    return pd.DataFrame(dict(success_rate=p, fail_rate=1.0 - p, shelf_life_p10=life[:, 0], shelf_life_p50=life[:, 1],
                             shelf_life_p90=life[:, 2]), index=rows.index)

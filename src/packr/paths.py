"""Where things live. Override the data folder with the PACKR_DATA_DIR environment variable."""
from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PKG_DIR / "reference"          # small curated tables shipped with the package
REPO_ROOT = PKG_DIR.parents[1]                 # valid for a source checkout / editable install
DATA_DIR = Path(os.environ.get("PACKR_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"                     # downloads you drop in by hand (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"         # tables produced by packr.build_tables (git-ignored)
MODELS_DIR = Path(os.environ.get("PACKR_MODELS_DIR", REPO_ROOT / "models"))   # trained bundles from packr.ml.train (git-ignored)

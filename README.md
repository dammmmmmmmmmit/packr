# Packr

AI-assisted food packaging recommendation: given a food's properties and its storage and transport
conditions, recommend a packaging structure (layers, barrier properties, MAP advice) and, later,
the probability that it protects the food for the target shelf life.

Approach (hybrid): a rule-based knowledge engine proposes and filters candidate structures, a
physics-based simulation scores them, and an ML model trained on the simulated outcomes generalises
the scoring. See [ROADMAP.md](ROADMAP.md) for the plan and the current gaps.

## Status

| Part | State |
|---|---|
| Knowledge engine (`packr.engine`) | Working. Thresholds are unvalidated starting points. |
| Material table | **Placeholder** values (`trust = placeholder`). Real values come from TRANSMAT / datasheets. |
| TRANSMAT reshape + unit conversion (`packr.transmat`, `packr.units`) | Built and tested on fixtures. **Not yet run on the real file.** |
| Spoilage sensor dataset loader and audit (`packr.spoilage`) | Built, tested on synthetic data. Not yet run on the real file. |
| Commodity table (48 foods) | Moisture and fat from USDA FoodData Central for 46 foods (`packr.fdc`); simulator parameters still assumed. |
| Shelf-life simulation / success rate (`packr.sim`) | First pass built, **uncalibrated** (every parameter assumed). |
| Train / test datasets (`data/datasets/`) | 85,100 train + 11,000 test rows, split by commodity. |
| Data-sufficiency study | Done: rows are enough, commodities are not (see `docs/IMPLEMENTATION_SPEC.md` section 6). |
| ML surrogate (`packr.ml`) | Built (phase 5): LightGBM success-rate and shelf-life heads with the commodity profile as features. Metrics in `reports/model_card.md`. |
| Optimiser (`packr.optimize`) | Built (phase 6): weighted score over structure, gauge and pack geometry; top 3 re-checked with a 1,000-trial simulator run. |
| API (`packr.api`) | Built (phase 7): `POST /recommend`, `GET /commodities` (FastAPI). Frontend not started. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # 183 tests
python -m packr.engine              # run the chips and strawberry demos
python -m packr.ml.train            # tune + fit + score the ML surrogate (writes reports/model_card.md, models/)
uvicorn packr.api:app --reload      # API on http://127.0.0.1:8000 (docs at /docs)
python -m packr.engine --scenario my.json --json     # one scenario from a file, JSON out
```

Scenario file:

```json
{"commodity": "Potato Chips", "moisture_pct": 2, "fat_pct": 33, "ph": 6.0,
 "shelf_life_days": 180, "temp_c": 25, "rh_pct": 60, "storage": "Ambient", "transport": "Normal"}
```

Build the cleaned data tables once the raw downloads are in `data/raw/` (see `data/README.md`):

```bash
python -m packr.build_tables --raw data/raw --out data/processed   # also writes data/processed/audit.txt
```

## Layout

```
src/packr/
  commodities.py    commodity master table (48 foods, simulator parameters, trust flags)
  sim/              physics.py (Monte Carlo simulator), dataset.py (train/test), sufficiency.py (is there enough data)
  ml/               features.py, train.py (grouped-CV tuning, model card), predict.py (load bundle, predict)
  profile.py        commodity profile: table lookup, user overrides, nearest-neighbour estimate for unknown foods
  optimize.py       phase 6 scoring and search over structures, gauge and pack geometry
  api.py            FastAPI app
  units.py          permeability unit conversion (barrer, um/kPa, mil/100in2 ...), OTR/WVTR, Q10
  materials.py      material table loader (bulk polymers vs fixed-barrier layers)
  transmat.py       TRANSMAT: parse entities, repair encoding damage, normalise permeabilities
  spoilage.py       spoilage sensor dataset loader, summary, informativeness check
  build_tables.py   raw downloads -> processed tables + audit report
  engine/           Scenario -> risks -> requirements -> candidate structures -> ranked recommendation
  reference/        small curated tables shipped with the package
tests/              pytest suite
notebooks/          engine_demo.ipynb
data/               raw/ and processed/ (git-ignored), README.md
```

## Conventions

* OTR is cm3/(m2 day) at a 1 atm oxygen difference; WVTR is g/(m2 day) at 38 C and 90 % RH.
* Permeability coefficients are stored as cm3 mm/(m2 day atm) (gas) and g mm/(m2 day kPa) (vapour),
  so OTR/WVTR scale as 1/thickness for homogeneous films. Coated, metallised, foil and perforated
  layers are `fixed`: their barrier does not scale with thickness.
* An unrecognised unit raises an error and is reported in the audit. It is never guessed.

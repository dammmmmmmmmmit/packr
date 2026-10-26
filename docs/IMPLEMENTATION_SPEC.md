# Packr: implementation spec (handover to Claude)

Owner: Adi. Source plan: *Proposed Methodology, AI-Powered Intelligent Packaging Recommendation System* (8 phases).
This file is written to be handed to an AI coding assistant. Read sections 0 to 3 first, then do the tasks in section 12 in order.
Repo: this project (`packr`, Python 3.11+). Run `pip install -e ".[dev]"` then `pytest` (183 tests should pass).

---

## 0. Ground rules for the implementer

1. **Nothing in the simulator is calibrated, and every simulator parameter is an assumption.** Files and columns carry a `trust` field. Never
   upgrade a `trust` value without a cited source (DOI, standard, datasheet). Do not "tune until the test passes".
2. **Splits are by commodity, never by row.** See section 5. Any new dataset code must keep the leakage guards in `sim/dataset.py`.
3. **Report two different things separately:** how well the ML model reproduces the simulator, and how well the simulator matches reality.
   Only the second is evidence about packaging. At present the second has almost no evidence (section 9).
4. Units: OTR cm3/(m2 day) at 23 C, 1 atm O2 difference. WVTR g/(m2 day) at 38 C / 90 % RH. Gas permeability cm3 mm/(m2 day atm); vapour g mm/(m2 day kPa).
   Unknown units raise `UnitError`; never guess.
5. Frozen storage is not modelled (inputs accepted, unused). Do not claim support for it.
6. Keep every success-rate output labelled "simulation-based estimate" in the UI and API (plan, phase 8: state assumptions clearly).

---

## 1. Objective (from the plan)

Given a food commodity and its storage and transport conditions, recommend packaging material, layer structure and specifications, and report a
**success rate and fail rate** against the desired shelf life, traceable to the rule engine and the simulation that produced it.

Inputs (plan phase 1): commodity, moisture %, fat %, pH, respiration rate, desired shelf life, temperature, RH, storage type (Ambient / Chilled / Frozen),
transport (Normal / Harsh).
Outputs (plan phase 7): number of layers, material per layer, OTR, WVTR, thickness, sealability, strength, MAP suitability, success rate, fail rate, predicted shelf life.

## 2. Plan phase vs repository state

| Plan phase | State | Where |
|---|---|---|
| 1 Requirements and knowledge | Done for produce and 12 dry snacks. Values mostly assumed | `commodities.py`, `reference/commodities_master.csv` |
| 2 Rule engine | Done. Thresholds unvalidated | `engine/` |
| 3 Physics simulation | **Built this round** (first pass, uncalibrated) | `sim/physics.py` |
| 4 Synthetic dataset | **Built this round**: 85,100 train rows, 11,000 test rows | `sim/dataset.py`, `data/datasets/` |
| 5 ML model | **Built** (T5): LightGBM success + shelf-life heads with commodity profile; acceptance met on held-out commodities (section 15) | `ml/`, `reports/model_card.md` |
| 6 Optimisation layer | **Built** (T6) | `optimize.py` |
| 7 Application | **API built** (T7); frontend not started | `api.py` |
| 8 Validation | Partly: two anchors run, results in section 9 | `sim/dataset.py::supplied_anchor, cookie_anchor` |

Real data now in `data/raw/` (2026-10-26): TRANSMAT (parsed; 114 of 165 permeability rows converted, values not yet tied to specific material grades), the spoilage sensor file (audited, unusable as packaging labels), FoodData Central (composition applied). Still missing: real material barrier values per grade, literature pairings for T3.
`data/README.md` lists the downloads.

---

## 3. Data assets

### 3.1 Commodity master: `reference/commodities_master.csv` (48 rows: 36 fresh produce, 12 dry snacks)

Built by `python -m packr.commodities` from: the 27-row produce seed, the supplied 25-food CSV pair (`reference/supplied_classifier.csv`,
`supplied_shelf_life.csv`, identical to the GitHub repo `srishanbangera-sys/Food_material_recommendation/backend/data` files), and numbers typed in code.

| Column group | Columns | Trust |
|---|---|---|
| Identity | `commodity_id`, `commodity`, `category` (`fresh_produce`/`dry_snack`), `subgroup` | n/a |
| Composition (model inputs) | `moisture_pct`, `fat_pct`, `ph` | moisture and fat: **USDA FoodData Central** for 46 of 48 (`reference/composition_fdc.csv`, built by `python -m packr.fdc`, approved 2026-10-26); still assumed for Extruded Puffs and Namkeen Sev. pH: supplied CSV for 25 foods, assumed otherwise (pH is not in FDC) |
| Respiration | `resp_ref_mgco2_kg_h` at `resp_ref_temp_c` = 10 C, `q10_resp` = 2.5 | seed value, or class midpoint (Very Low 5, Low 9, Medium 17, High 27, Very High 62). **The 10 C reference is an assumption**: the seed never recorded it |
| Storage | `ideal_temp_c`, `ideal_rh_pct`, `base_life_days` | seed, else medians of the supplied CSV. `base_life_days` = life at ideal temperature in an ordinary non-MAP pack |
| Produce simulator parameters | `o2_min_pct`, `co2_max_pct`, `crit_weight_loss_pct`, `ma_gain`, `transp_g_kg_d`, `q10_decay`, `fragility`, `chilling_sensitive` | **all assumed**, literature-informed orders of magnitude |
| Snack simulator parameters | `crit_aw`, `iso_b`, `ox_ref_days`, `ea_ox_kj`, `light_sens` | **all assumed** |
| Trust flags | `trust_composition`, `trust_respiration`, `trust_sim_params` | `trust_sim_params` is `assumed` on every row |

### 3.2 Materials: `reference/materials_placeholder.csv` (12 materials, `trust = placeholder`)

Unchanged. 89 candidate structures come from `engine.structures.build_candidates` (5 monolayers + 84 laminates). Replace with real values in task T1.
Row-level gauge is varied by a `gauge_scale` in [0.5, 2.0] that multiplies the thickness of `bulk` layers only (`sim/dataset.py::scaled`).

### 3.3 Datasets (`data/datasets/`, gzip CSV, regenerate with `python -m packr.sim.dataset --out data/datasets --seed 0`, about 40 s)

| File | Rows | What |
|---|---|---|
| `train.csv.gz` | 85,100 | 37 real train commodities x 500 rows (18,500) + 444 virtual commodities x 150 rows (66,600). 128 Monte Carlo trials per row |
| `test.csv.gz` | 11,000 | 11 held-out real commodities x 1,000 rows. 256 trials per row. No virtual commodities |
| `anchor_supplied.csv.gz` | 50 | the supplied shelf-life rows re-run through the simulator (not for training) |
| `anchor_cookie.csv.gz` | 3 | the three-film cookie study from ROADMAP section 6 (not for training) |
| `commodities_master.csv` | 48 | copy of the master table |
| `dataset_summary.json` | | seed and counts |

Held-out test commodities (fixed list `TEST_COMMODITIES`): Strawberry, Lettuce, Mango, Potato, Cucumber, Apple, Okra, Cauliflower, Banana Chips, Cream Biscuits, Roasted Peanuts.
Potato Chips stay in train because they are the headline use case. **After evaluation the production model must be retrained on all 48 commodities.**

**Row schema (66 columns)**

| Group | Columns |
|---|---|
| ids | `row_id`, `commodity`, `commodity_id`, `parent_id` (the real commodity a virtual one came from), `source` (`real`/`virtual`), `split`, `fold` (0 to 4 on train, -1 on test), `category` |
| Plan inputs | `moisture_pct`, `fat_pct`, `ph`, `resp_rate_mgco2_kg_h` (at the storage temperature, Q10-scaled), `target_days`, `temp_c`, `rh_pct`, `storage`, `transport` (+ `harsh` 0/1) |
| Package spec | `layers` ("PET/AlFoil/LDPE"), `n_layers`, `gauge_scale`, `otr23`, `wvtr38`, `thickness_um`, `light_barrier`, `seal`, `strength`, `cost_index`, `recyclable`, `breathable`, `mass_kg`, `area_m2_per_kg`, `perf_mm2_per_kg` (produce), `flush_o2_pct` (snacks) |
| Latent commodity parameters (not available from the plan inputs) | `resp_ref_mgco2_kg_h`, `resp_ref_temp_c`, `q10_resp`, `ideal_temp_c`, `base_life_days`, `chilling_sensitive`, `fragility`, `o2_min_pct`, `co2_max_pct`, `crit_weight_loss_pct`, `ma_gain`, `transp_g_kg_d`, `q10_decay`, `crit_aw`, `iso_b`, `ox_ref_days`, `ea_ox_kj`, `light_sens` |
| Labels | `success_rate`, `fail_rate`, `shelf_life_p10/p50/p90`, `p_fail_moisture`, `p_fail_oxidation`, `p_fail_o2_low`, `p_fail_co2_high`, `p_fail_weight_loss`, `p_fail_quality`, `x_o2_pct_p50`, `x_co2_pct_p50`, `n_trials` |
| Baseline | `rule_engine_pass` (1 if the rule engine's requirements accept this package) |

Label distribution (train): mean success rate 0.54 for dry snacks, 0.25 for fresh produce. Test: 0.59 and 0.18 (0.69 and 0.19 before the FDC composition). Produce is dominated by zeros because most
sealed barrier laminates are anaerobic for respiring produce; that is physics, not a bug.

### 3.4 Sampling design (why the rows look the way they do)

* Storage: produce 65 % Chilled (0 to 12 C, RH 80 to 98) / 35 % Ambient (15 to 38 C, RH 50 to 95); snacks 90 % Ambient, RH 30 to 90. Transport Harsh 30 %.
* Target shelf life: produce, drawn around the life achievable at that temperature (`base_life x q10_decay^(-(T - ideal)/10)` times exp(U(ln 0.3, ln 1.6))) for 85 % of rows, and a wide
  range (0.1x to 6x) for 15 % so impossible requests exist. Snacks: log-uniform 30 to 400 days.
* Package: 40 % of rows pick a structure the rule engine would accept, 60 % are uniform over the 89 candidates. For produce, 70 % of rows get a perforation area **designed**
  to hit a random in-pack O2 between about 6 and 20 %; without this almost every produce label was 0.
* Virtual commodities: each train commodity gets 12 siblings with log-normal or normal jitter on respiration, composition, life, tolerances and the simulator parameters
  (`make_virtual`). Used only to enlarge train; they are not independent evidence of new foods.

---

## 4. Simulator specification (`sim/physics.py`)

Vectorised NumPy; one call simulates many rows, `n_trials` draws each. `success_rate` = share of draws that reach `target_days` with no hard failure.
Shelf-life columns are percentiles over draws. All constants are in the `Constants` dataclass.

**Draws per trial (shared):** temperature = set point + harsh offset (1.5 K) + N(0, 1.5 K normal / 3.5 K harsh); RH + N(0, 4); OTR and WVTR tolerance each lognormal sigma 0.15;
seal leak with probability from seal rating (1: 30 %, 2: 10 %, 3: 4 %, 4: 2 %, 5: 1 %; doubled if harsh) giving a pinhole of lognormal area (median 1e-6 cm2, sigma 1.5);
transport damage (3 % normal, 12 % harsh, scaled by fragility and (6 - strength)/3) giving a pinhole of median 1e-4 cm2.
Film permeability scales with Arrhenius factors: O2 Ea 40 kJ/mol (ref 23 C), vapour Ea 25 kJ/mol (ref 38 C). Holes pass gas by diffusion: G = D * 86400 * area / length
(D_O2 0.20, D_vapour 0.25 cm2/s; length 25 um for perforations, 50 um for seal leaks).

### 4.1 Dry snacks (failure: texture loss OR rancidity)

* Moisture: linear isotherm, aw = M/b. `dM/dt = (C_wv / (Ws b)) (b RH - M)`, with C_wv = (WVTR/delta_p38) x area x p_sat(T) + pinhole term. Time to critical moisture
  `Mc = b x crit_aw`: `t = ln((Me - M0)/(Me - Mc)) / K`, infinite if `Me = b RH <= Mc`, zero if already above.
* Oxidation: `t_ox = max(t_intrinsic, (Q_crit - O2_headspace) / ingress)`. `t_intrinsic = ox_ref_days x exp(Ea/R (1/T - 1/298.15))` x lot variability (lognormal 0.25) / light factor
  (`1 + light_sens (1 - light_barrier) U(0,1)`). `Q_crit` = 1 cm3 O2 per g fat. Headspace 2.5 cm3/g at `flush_o2_pct`. Ingress = (OTR_T x area + pinhole) x 0.209.
* Life = min(moisture, oxidation) x 0.9 if damaged.

### 4.2 Fresh produce (failure: anaerobic, CO2 injury, weight loss, or quality decay)

* Respiration: `R = R_ref x lognormal(0.25) x Q10^((T - T_ref)/10)`, Q10 ~ N(q10_resp, 0.2); 13.1 cm3 CO2 per mg CO2 per day; RQ ~ N(1, 0.1).
* Steady-state gas balance, solved by 40-step bisection per draw: `G_O2 (0.209 - x) = R_O2 x/(Km + x) / norm`, Km ~ U(1, 3) % O2. CO2: `x_CO2 = R_O2 RQ / G_CO2 + 0.0004`,
  `G_CO2 = beta G_film + 0.8 G_holes`, beta ~ N(4, 0.6). **Steady state is assumed; transients and initial gas flush are ignored.**
* Hard failures: O2 below `o2_min_pct` x U(0.8, 1.2); CO2 above `co2_max_pct` x U(0.8, 1.2). Life is multiplied by 0.3 when violated.
* Water: series resistance of product transpiration (`transp_g_kg_d` x mass x p_sat ratio to 20 C) and the pack; `loss = k c (0.99 - RH)/(k + c)`; weight-loss time =
  `crit_weight_loss_pct x mass / loss`.
* Quality life: `base_life / q10_decay^((T - ideal)/10)`, chilling-injury penalty below `ideal - 2` C, modified-atmosphere benefit `1 + ma_gain x benefit`
  (benefit = 0.7 x O2 reduction toward 3 % + 0.3 x CO2 up to 5 %), lot variability lognormal 0.15, condensation penalty when in-pack RH > 97 %, damage 0.8x.
* Life = min(quality, weight-loss time).

### 4.3 Known simulator limits

No humidity dependence of film permeability (EVOH, PA, paper); no ethylene; no microbial model; no pack-level sorption; produce at steady state only; snack linear isotherm;
a single `ma_gain` curve per commodity; no frozen storage; no corrugated or mesh containers (the supplied CSV lists them; they are not representable except as very large perforation).
Options B1 to B5 (4.4) address some of these, but are off by default and inert until their reference tables hold sourced rows.

### 4.4 Structural options (T3, off by default)

Selected by `Constants` fields marked `metadata={"structural": True}` (`Constants().options()`, `Constants().is_v1`). The default is v1, verified bit-identical
to the pre-T3 code on 4,000 saved dataset rows. When an option is on, `simulate` adds a `sim_model` column naming the options and, per row, whether it fell back
to v1 because no sourced parameters were found. `simulate(..., extra=True)` adds `EXTRA_COLS` (per-mechanism life percentiles, a_w0, PV at target, etc.).

| Option | Field | Physics | Parameters from | Without sourced rows |
|---|---|---|---|---|
| B1 GAB isotherm | `isotherm="gab"` | `M = m0 C K aw / ((1-K aw)(1-K aw+C K aw))`; a_w0 by inverting the quadratic; time to critical moisture by Gauss-Legendre quadrature in `u = -ln(RH - aw)` | `reference/sorption_params.csv` (m0, C, K, crit_aw, crit_basis) or per-row `gab_*` / `crit_moisture_db` columns | linear isotherm |
| B1e empirical curve | `isotherm="gab_empirical"` | same as B1, but only rows with trust `empirical_fit...` (curves fitted to storage-trial (moisture, aw) pairs by `sim/sorption_fit.py`); never used outside `valid_aw_min..valid_aw_max`: a trial whose path leaves the range uses v1, and `sim_model` records the share | `sorption_params.csv` empirical rows (Banana Chips only) | linear isotherm |
| B2 PV oxidation | `oxidation="pv"` | headspace O2 balance `V dp/dt = G(0.209 - p) - A f(p)`, `f = p/(km + p)`; `dPV/dt = kpv f(p)/f(0.209)`; 1 mEq PV = 11.2 cm3 O2; end point `pv_endpoint_meq_kg` (10) | `reference/oil_stability.csv` keyed by master `oil_type` (empty) | v1 oxidation |
| B3 RH-dependent OTR | `rh_dependent_otr=True` | RH at each layer from cumulative vapour resistance; layer OTR x multiplier(RH); laminate OTR as a series sum | `reference/rh_permeability.csv` | multiplier 1 |
| B4 temperature excursions | `temp_history="excursions"` | Poisson excursions (rate 1 / 3 per 30 d normal / harsh; lognormal duration, median 0.5 / 1 d; amplitude N(6 / 10, 3) K, max 8 per trial); rates time-averaged (Arrhenius-equivalent temperature) | `Constants.exc_*`, **ASSUMED** | n/a (always active) |
| B5 transient produce gas | `produce_gas="transient"` | semi-implicit O2/CO2 ODE from air over 8 tau (free volume 1000 cm3/kg); quality life consumed during the transient is counted | `Constants.produce_free_volume_cm3_per_kg`, **ASSUMED** | n/a |

Limits: B5 judges O2/CO2 violations at steady state only; B4/B5 constants are assumed; B2 needs oil types per snack.

---

## 5. Leakage rules (already enforced; keep them)

1. A commodity and all its virtual copies are entirely in train or entirely in test (`parent_id`). Asserted in `build()`.
2. Virtual commodities are built only from train commodities. The test set has real commodities only.
3. Cross-validation folds are by parent commodity (`fold`); never use plain row-wise KFold.
4. Anchors are never used to fit or to pick hyperparameters.
5. Do not select features or thresholds by looking at the test file. Tune on `fold`, report on test once.

---

## 6. Is there enough data? (study in `sim/sufficiency.py`, results in `reports/sufficiency.json`)

Acceptance criteria were fixed in code before the run: held-out MAE of success rate <= 0.10; at least 30 % better than the best baseline; error plateau (going from 30 to 37 commodities
improves MAE by <= 5 %); AUC >= 0.90 for "success >= 0.5"; median |ln(pred/simulated)| of shelf life <= 0.30. Model: LightGBM (400 trees, 63 leaves, cross-entropy objective for success rate),
22 features = the plan inputs plus the package spec (no latent parameters).

| Check | Result | Criterion | Pass |
|---|---|---|---|
Re-run 2026-10-26 after the FDC composition (T4); the first-run figure is in brackets. First-run reports: `reports/archive_pre_fdc/`.

| Check | Result | Criterion | Pass |
|---|---|---|---|
| MAE of success rate, 11 unseen commodities | **0.178** (0.155) | <= 0.10 | **no** |
| Improvement over best baseline (category mean 0.329) | 46 % (53 %) | >= 30 % | yes |
| AUC (success >= 0.5) | 0.905 (0.936) | >= 0.90 | yes |
| Median log error of shelf life | **0.70** (0.59) | <= 0.30 | **no** |
| Plateau in number of commodities | 5.1 % gain, noisy (14 %) | <= 5 % | **no** |

Other measurements:

* Baselines (MAE): global mean 0.370, category mean 0.329, rule-engine pass/fail 0.368 (AUC only 0.621). The rule engine alone is a weak predictor of the simulated outcome.
* **Oracle test:** the same model given the latent commodity parameters reaches MAE **0.082** (0.073), AUC 0.981, shelf-life log error 0.26, so all three accuracy criteria are met. The shortfall is
  therefore **missing commodity-level information, not model capacity or row count.**
* Rows: MAE 0.190 at 1 % of rows (851), 0.177 at 10 %, 0.174 at 30 % (25,530 rows), 0.179 at 85,100: flat after a few thousand rows. **Row count is sufficient.** More simulated rows will not help.
* Commodities: MAE vs number of training commodities is non-monotonic (real + virtual: 0.185, 0.208, 0.178, 0.178, 0.187, 0.178 for 6, 12, 18, 24, 30, 37 commodities; three random subsets each).
  The spread between subsets (up to about 0.04) is as large as the trend, so **37 real commodities cannot show a clear learning curve**. Virtual copies helped at the full set (0.178 vs 0.189
  real rows only) but not consistently.
* Per-commodity MAE on test: Roasted Peanuts 0.081, Cauliflower 0.118, Cucumber 0.122, Apple 0.133, Potato 0.141, Mango 0.158, Strawberry 0.186, Cream Biscuits 0.209, Banana Chips 0.237, Okra 0.280, Lettuce 0.289.
* Grouped 5-fold CV on train commodities: mean MAE 0.167, sd 0.036 (range 0.115 to 0.219). Expect the final number on a new commodity to be anywhere in that range.
* Category extrapolation fails: a model trained only on produce has MAE 0.59 on snacks; trained only on snacks, 0.20 on produce. **Treat each category as needing its own training commodities.**
* Coverage: median nearest-neighbour distance (standardised composition, respiration, life, Q10) from a test commodity to its category's train commodities is 0.64, against 0.50 leave-one-out
  within train. The test commodities are somewhat further from train than train commodities are from each other, so the test is mildly conservative.
* Monte Carlo label noise floor on the test file: MAE about 0.006, negligible.

**Verdict.** Rows: more than enough. Commodities: not enough for the model to be trusted on a food it has not seen, if it sees only the plan's ten inputs. The plan's inputs do not determine
the commodity-specific limits (oxygen floor, CO2 tolerance, critical water activity, base life), and 37 foods are too few to infer them from moisture, fat, pH and respiration alone.
Accuracy on foods already in the commodity table should be close to the oracle figure.

**Design decision this forces (section 7.2):** the model should take a commodity profile, looked up from the commodity table by name, not only the raw plan inputs. For a new food the
user supplies, or the system estimates from its nearest neighbours, the profile; the response then carries a "profile estimated" flag and a wider interval.

---

## 7. Phase 5: ML model specification (to build)

### 7.1 Targets and heads

1. `success_rate` in [0, 1]; fail rate = 1 - success. LightGBM `objective="cross_entropy"` (accepts fractional labels). Optionally a second head per failure mode (`p_fail_*`) to explain a low rate.
2. `log(shelf_life_p50)`; also `p10` and `p90` as quantile-style heads or two extra regressors, for an interval.
3. Optional: FT-Transformer (plan phase 5 alternative) on the same features, compared on the same folds. Do it only after the GBT baseline is reported.

### 7.2 Features

* Always: the 22 columns in `sufficiency.FEATURES` (category flag, composition, respiration, log target, temperature, RH, storage, transport, log OTR, log WVTR, light, seal, strength, thickness,
  layers, breathable, mass, area per kg, log perforation, flush O2).
* **Add the commodity profile**: the `LATENT` list in `sufficiency.py` (base life, Q10 decay, O2 min, CO2 max, critical weight loss, MA gain, transpiration, critical aw, isotherm slope, oxidation reference days,
  oxidation Ea, light sensitivity, ideal temperature, fragility). Missing values (a snack has no `o2_min_pct`) are NaN; LightGBM handles them.
* For an unseen food: profile source order = user-entered values, then the average of the 3 nearest commodities in the same category (standardised moisture, fat, log respiration, log life), flagged as estimated.

### 7.3 Training protocol

* Tune hyperparameters with grouped CV on `fold` (train only). Fix the model, refit on all train, score `test.csv.gz` once.
* Report per-commodity, per-category and overall MAE, AUC, calibration curve (predicted vs mean simulated success in 10 bins), and the 4 baselines (global mean, category mean, rule-engine pass, oracle).
* Final deliverable model: refit on train **plus** test commodities, with a new virtual expansion of the test commodities (regenerate with `TEST_COMMODITIES = []` after freezing hyperparameters).
* Acceptance (tightened from section 6, with the profile features): held-out MAE <= 0.10, AUC >= 0.90, shelf-life log error <= 0.30, calibration error per bin <= 0.10. If not met, say so; do not change the criteria.

### 7.4 Explainability (plan expected outcome)

Every recommendation returns: the rule-engine trace (`Requirements.trace`), the failure-mode breakdown (`p_fail_*` from the simulator, not from the ML model) and, for the chosen package, a direct
simulator run with 1,000 trials (cheap) so the displayed success rate can be the simulator's, with the ML model used for ranking and for fast search over many candidates.

---

## 8. Phase 6: optimisation layer (to build)

Candidates: all structures passing hard filters (seal and strength minimums from the rule engine; `breathable` requirement for very high respiration is a preference, not a filter).
Score each candidate (the structure's gauge scale and pack geometry are search variables inside bounds from section 3.4):

`score = w_s * P_success + w_c * (1 - cost_norm) + w_g * sustain + w_m * manufacturability`, default weights 0.6 / 0.15 / 0.15 / 0.10, user-adjustable.

* `cost_norm`: `cost_index` (currently unitless and ignores thickness, replace by cost per m2 at the chosen gauge, task T6).
* `sustain`: 1 if single polymer family (`recyclable`), else 0 for now; replace by a recyclability rating when data exists.
* `manufacturability`: seal rating / 5 averaged with strength / 5.
* Hard rule: never recommend a package with P_success < 0.5 as "recommended"; return the best available with a "no package meets the target" note.
Return top 3 with their scores and the weight vector used.

## 9. Phases 7 and 8

### 9.1 API (FastAPI, `packr/api.py`)

`POST /recommend` body = a `Scenario` plus optional `commodity_profile` overrides and `weights`. Response: top-3 packages (layers, thickness, OTR, WVTR, light, seal, strength, MAP advice),
`success_rate`, `fail_rate`, `shelf_life_p10/p50/p90`, failure-mode breakdown, rule trace, `profile_source` (`table`/`user`/`estimated`), `disclaimer` ("simulation-based estimate; not validated against
measured spoilage data"). `GET /commodities` lists the table. Frontend (Flutter or web) is out of scope until the API passes its tests.

### 9.2 Validation results so far (do not overstate)

* **Supplied shelf-life CSV (50 rows).** Its 28 non-MAP rows: simulated median / observed = 1.33, Spearman 0.85. This is **not independent**: for foods that appear only in that CSV, `base_life_days`
  was taken from those same rows. Its 22 "MAP High Barrier" rows: simulated / observed median 0.22. They list sealed films with OTR 6 to 15 for strawberry, spinach, broccoli etc., with 12 to 21 day lives.
  At that OTR the simulated pack is anaerobic. Either those rows describe packs with perforations or higher-permeability films, or the numbers are wrong. **Do not use the MAP rows as ground truth.**
  Note also that the two supplied CSVs look hand-authored: the earlier audit found conflicting labels (best possible accuracy about 0.56 on the classifier labels).
* **Cookie study (three films, 22 C, 50 % RH, 6 months).** Observed: PP (OTR 1307, WVTR 5) and PVdC/PP (OTR 21.8, WVTR 4.2) compromised, metallised PP (31.2, 0.4) acceptable. With the assumed
  `Cream Biscuits` (`crit_aw` 0.55; moisture now 2.2 % from FDC) the simulator passes **all three** (success 0.97, 0.97, 1.00): **the direction is not reproduced**, matching the earlier finding that the engine's medium-moisture tier is too lenient.
  Sweeping `crit_aw` (FDC composition): 0.30 to 0.35 gives (0, 0, 1), the observed direction; 0.40 gives (0.03, 0.04, 1.0); 0.45 gives (0.24, 0.33, 1.0); 0.50 gives (0.74, 0.79, 1.0); 0.55 gives (0.96, 0.97, 1.0). Caramelised cookies plausibly have a lower
  critical water activity than the assumed biscuit. This is a hypothesis from one study with unknown composition and pack geometry. Confirm from the paper (Hasselt University handle 1942/33777) before changing the table.
* Phase 3 "done when" criterion (reproduce pass/fail direction of at least 3 literature pairings) is **not met** (0 of 1 verified so far).

---

## 10. Interfaces to keep stable

`packr.engine.recommend(scenario, materials=None, top_k=3)`, `packr.sim.physics.simulate(df, n_trials, seed)`, the dataset schema in 3.3, `sufficiency.FEATURES` and `LATENT` names,
`commodities_master.csv` columns. If you must change one, update the schema table here and bump the dataset seed note in `dataset_summary.json`.

T3 additions (backward compatible): `simulate(df, n_trials, seed, c=CONSTANTS, chunk=..., extra=False)`; `sim_model` column only when a structural option is on;
`commodities_master.csv` gained `oil_type` and `oil_type_doi` (both empty). The master's sha256 therefore no longer matches `commodities_master_sha256` in
`dataset_summary.json`; composition is unchanged and the datasets were not regenerated (awaiting approval).

## 11. Verify first (all assumed, ranked by how much they move results)

1. Snack `crit_aw`, `iso_b`, `ox_ref_days` (12 snacks) from Labuza-type isotherm and shelf-life literature or measured data. They set most snack labels.
2. Produce `o2_min_pct`, `co2_max_pct`, `ma_gain`, `q10_decay` per commodity (modified-atmosphere tolerance tables: Kader, Saltveit, Mir and Beaudry).
3. Respiration reference temperature (assumed 10 C) and values; check all 36 against USDA Handbook 66.
4. `base_life_days` for the 9 foods that exist only in the supplied CSV (taken from its own rows).
5. Moisture and pH for 11 produce and all snacks (USDA FoodData Central).
6. Material barrier values (all 12 are placeholders).
7. Simulator constants in `Constants`: leak probabilities and pinhole areas, Ea values, headspace volume, `qcrit_ml_per_g_fat`, damage rates, `transp_g_kg_d`.
Since T3 the order is measured, not guessed: `reports/sensitivity.md` (Morris). The look-up list in the T3 log entry (section 15) follows it.

8. The sampling design bias: because 40 % of structures are pre-filtered by the rule engine and 70 % of produce perforations are designed, the data distribution is not the distribution of real user requests.

---

## 12. Task list (do in order; each has a done-when)

| # | Task | Done when |
|---|---|---|
| T0 | Install, run `pytest`, run `python -m packr.sim.dataset --quick --out /tmp/q` | 82 tests pass, quick build works |
| T1 | Phase 1 real data: run `python -m packr.build_tables` on `goldenTRANSMAT-all.tab` and `Dataset.csv` (user supplies); solve value-to-material association; build `materials_from_transmat.csv` with DOI per row; replace placeholder rows | at least 20 real material rows with DOIs; trust column updated; audit shows >= 80 % permeability rows converted |
| T2 | Fill composition from FoodData Central; verify items 3 to 5 of section 11; set `trust_*` accordingly | no `assumed` composition left; changes listed in a CHANGELOG section |
| T3 | Calibrate the simulator (phase 8): fix `crit_aw` and snack parameters against the cookie study and any other pairings found; add each pairing as a row in a new `reference/literature_pairings.csv` with DOI, composition, film, conditions, outcome | simulator reproduces direction of >= 3 independent pairings; none were used to fit the parameter they test |
| T4 | Regenerate datasets (`--seed 0`), re-run `python -m packr.sim.sufficiency`; update section 6 numbers | report committed under `reports/` |
| T5 | Build the ML model (section 7): `packr/ml/train.py`, `predict.py`, saved with `joblib` plus a model card (`reports/model_card.md`) containing the protocol and metrics | acceptance in 7.3 reported honestly; tests for feature schema and monotonicity (higher WVTR must not raise snack success on average) |
| T6 | Optimisation layer (section 8) in `packr/optimize.py`; replace `cost_index` with cost per m2 | top-3 ranked output with scores; unit tests on ordering |
| T7 | API (section 9.1) with request validation reusing `Scenario` validation, and tests with `fastapi.testclient` | endpoint tests pass; response includes disclaimer and `profile_source` |
| T8 | FT-Transformer comparison (optional) on the same folds | one table comparing GBT vs FT-T on held-out commodities |
| T9 | Frontend (Flutter or web) | out of scope until T7 is done and the user confirms the platform |

## 13. Open questions for Adi

1. Commodity scope: produce plus dry snacks (built), or produce only? (The dataset has both.)
2. Can you supply `goldenTRANSMAT-all.tab` and `Dataset.csv`, or any measured shelf-life data from your department? One real measured pairing is worth more than 10,000 more simulated rows.
3. Team size and the submission or demo date (decides whether T7 and T9 are in scope).
4. For the cookie result: do you have the paper's cookie composition and pack size?

## 14. Reproducing every number in this file

```bash
pip install -e ".[dev]"
python -m packr.fdc --raw data/raw --out data/processed   # FDC composition audit (needs the FDC downloads; result shipped in reference/composition_fdc.csv)
python -m packr.commodities                          # rebuild commodity master (48 rows, FDC moisture and fat applied)
python -m packr.sim.dataset --out data/datasets --seed 0
python -m packr.sim.sufficiency --data data/datasets --out reports     # about 10 minutes
python -m packr.ml.train --data data/datasets --out reports --models models                 # phase 5 model card
python -m packr.sim.dataset --out data/datasets_all --seed 0 --no-test
python -m packr.ml.train --final --data data/datasets_all --params reports/ml_metrics.json --models models
python -m packr.sim.sensitivity --data data/datasets --out reports   # Morris ranking (needs SALib, dev extra), a few minutes
python -m packr.sim.sorption_fit --write                             # empirical banana-chip GAB curve -> reference/sorption_params.csv
python -m packr.sim.calibrate                                        # scores reference/literature_pairings.csv -> reports/calibration.md (about 1 min)
pytest
```
Numbers in section 6 come from `reports/sufficiency.json` (seed 0, LightGBM 4.7.0). Re-running with another seed will move individual figures by up to about 0.03 MAE (see the CV spread).

## 15. Implementation log

### 2026-10-26: T0, T5, T6, T7

| Task | State |
|---|---|
| T0 | Done. 82 original tests pass; quick build works. |
| T1, T2, T3 | **Blocked on inputs**: `goldenTRANSMAT-all.tab`, `Dataset.csv`, FoodData Central lookups and literature pairings are not in the repo. No `trust` value was changed. |
| T4 | Done later the same day, after T2 approval (see the next entry). |
| T5 | Done: `packr/ml/{features,train,predict}.py`, `reports/model_card.md`, `reports/ml_metrics.json`, `models/packr_gbt_eval.joblib` (train commodities only) and `models/packr_gbt.joblib` (all 48, regenerated with `--no-test`). |
| T6 | Done: `packr/optimize.py`. |
| T7 | Done: `packr/api.py` (`POST /recommend`, `GET /commodities`), tests with `fastapi.testclient`. |
| T8, T9 | Not started. |

**T5 results** (seed 0 data, test scored once after grouped-CV tuning; full tables in `reports/model_card.md`). These measure agreement with the simulator, not with reality.

| Check (7.3) | Value | Limit | Pass |
|---|---|---|---|
| MAE of success rate, 11 held-out commodities, profile from the table | 0.067 | <= 0.10 | yes |
| AUC (success >= 0.5) | 0.988 | >= 0.90 | yes |
| Median log error of shelf life p50 | 0.21 | <= 0.30 | yes |
| Worst calibration bin (>= 30 rows) | 0.056 | <= 0.10 | yes |

* New-food case (profile estimated from the 3 nearest train commodities): MAE **0.146**, AUC 0.935, shelf-life log error **0.55**. That **fails** the criteria; only foods in the commodity table get the figures above. The API flags estimated profiles and reports a success range over the neighbour profiles.
* Plan inputs only (no profile): MAE 0.165, matching section 6. Grouped CV with the chosen parameters: MAE 0.087, sd 0.028 (folds 0.064 to 0.139).
* Worst commodity: Roasted Peanuts, MAE 0.141 (under-predicted, 0.52 vs 0.66). The snack shelf-life log error (0.32) is above 0.30 on its own; produce (0.18) carries the average.
* The chosen `num_leaves = 127` is the edge of the grid. A larger grid can be tried **on folds only**; the test file must not be used to choose it.
* The success head is constrained monotone non-increasing in `log_target_days` (a property of the simulator). WVTR monotonicity for snacks is tested, not constrained: doubling WVTR lowers mean predicted snack success from 0.630 to 0.586.

**Design decisions taken (change them if you disagree)**

* `FEATURES`, `LATENT`, `featurise` moved to `packr/ml/features.py` so the API does not import matplotlib; `sufficiency` re-exports the same objects (section 10 names unchanged).
* `sim/dataset.py`: `build(test_commodities=...)` / `--no-test` added for the final refit; the perforation-design formula became `perforation_cm2_for_o2` (output verified byte-identical); `candidate_table` gained `cost_bulk` / `cost_fixed`.
* Cost (T6): `cost_rel_per_m2 = gauge_scale x cost(bulk layers) + cost(fixed layers)`, using the existing relative `cost` column as cost per m2 at nominal gauge. Still unitless and **placeholder**; no prices were invented.
* Optimiser: the ML model ranks every variant (structure x 5 gauges x 3 areas x {air, N2 flush} or {no perforation, 4 designed O2 levels}); each structure keeps its best variant; the best 10 are re-run in the simulator with 1,000 trials, and that simulated rate is shown and used for the final rank and the >= 0.5 "recommended" rule. Packages meeting the target always rank above those that do not; if none meets it, the rest are ordered by success rate. Without a model file the search falls back to a 64-trial simulator run.
* Default pack: 0.1 kg (snack) / 0.5 kg (produce); the area per kg is searched unless the request fixes it. Area does not enter the cost term (cost is per m2 as specified), so the smallest area usually wins for snacks.
* Frozen storage: `/recommend` returns rule-engine structures with `supported: false` and no success rate.
* An unknown food's category is guessed from moisture (<= 15 % means dry snack) unless the request gives `category`.

**Known rough edges**: the candidate generator yields mirrored structures (MetPET/MetBOPP/CPP and MetBOPP/MetPET/CPP), which can both appear in a top 3; best variants sit on the search-grid bounds (gauge 0.5, area 0.6) because the placeholder cost and barrier values reward that.

### 2026-10-26 (later): T1 and T2 on the real raw files, awaiting approval

Nothing downstream changed: `commodities_master.csv`, `materials_placeholder.csv`, every `trust` value, the simulator, datasets and models are untouched.

* T1: `python -m packr.build_tables` now reads `goldenTRANSMAT-all.csv`. Parser fixes (real Unicode ×, −, ±, °; scientific notation; table-header
  multipliers) and dimensional-analysis unit parsing: 165 permeability rows, **114 converted (69 %)**, every other row listed with its reason in
  `data/processed/audit.txt` (22 are permeances / transmission rates, rejected by rule). Temperatures found: 58 (was 0).
  `packr.associate` writes `materials_from_transmat.csv`: **29 rows, 6 DOIs**, association = single-material paper with the variant (neat vs
  composite) never resolved. No paper names exactly one packaging term. No placeholder row is replaced.
* T2: `python -m packr.fdc` writes `composition_from_fdc.csv` (45 of 48 matched by the explicit alias table in `packr/fdc.py`),
  `composition_disagreements.csv` (9 values > 3 pp, e.g. Avocado fat 0.3 -> 20.3) and `commodities_master_proposed.csv` + `_diff.csv`.
  pH unchanged (not in FDC).

### 2026-10-26 (approval round): T2 applied, T4 done

Adi approved: the FDC composition table; Salted Crackers = snack-type crackers; the relaxed TRANSMAT association kept as reference only;
O2/CO2 in mass units stays rejected (`family_mismatch`).

* `reference/composition_fdc.csv` (46 rows) is applied by `commodities.build_master`; `commodities_master.csv` rebuilt and identical (within 0.01 rounding)
  to the approved proposal. Only `moisture_pct`, `fat_pct`, `trust_composition` changed. Old table: `reports/archive_pre_fdc/commodities_master_pre_fdc.csv`.
* Datasets regenerated (`--seed 0`, same split and row counts); `dataset_summary.json` now records `commodities_master_sha256`.
* Section 6 re-run (numbers above). Phase 5 model retuned on folds, scored once, refit on all 48:

| Check (7.3) | Before FDC | After FDC | Limit | Pass |
|---|---|---|---|---|
| MAE of success rate, held-out commodities | 0.067 | **0.076** | <= 0.10 | yes |
| AUC | 0.988 | 0.983 | >= 0.90 | yes |
| Shelf-life log error | 0.210 | 0.231 | <= 0.30 | yes |
| Worst calibration bin | 0.056 | 0.056 | <= 0.10 | yes |
| New-food case (estimated profile), MAE | 0.146 | 0.172 | | fails |
| Grouped CV, chosen parameters | 0.087 (sd 0.028) | 0.073 (sd 0.011) | | |

* **Finding for T3 (not tuned):** with FDC moisture 4.3 %, the assumed `iso_b` 0.09 puts Banana Chips at an initial a_w of 0.50, exactly the assumed
  `crit_aw` 0.50, so 61 % of simulated packs fail on moisture at once and the model's Banana Chips MAE rose from 0.05 to 0.19 (snack shelf-life log error 0.59).
  Real chips sold at 4.3 % moisture are crisp, so `iso_b` or `crit_aw` for banana chips is wrong. Instant Noodles are close as well (a_w 0.54 vs 0.60).
  Roasted Almonds sat exactly at its limit under the old assumed moisture and no longer does. Fix only from sorption-isotherm data (spec 11, item 1).

### 2026-10-26 (T3 round): sensitivity, structural options, calibration machinery

No parameter value, trust value, dataset, split or model changed. 207 tests pass (16 new in `tests/test_structural.py`, 8 in `tests/test_calibrate.py`).

* **Part A**: `packr/sim/sensitivity.py` (Morris, SALib; 300 real train rows per category, 64 trials, 40 trajectories, common random numbers;
  a re-run with another seed and 80 trajectories gave Spearman 0.99 on the ranking). Report: `reports/sensitivity.md` / `.json`.
  Dry snacks: `iso_b` > `crit_aw` > `ox_ref_days` > moisture (sourced) > `qcrit_ml_per_g_fat` > `headspace_cm3_per_g` > `ea_wv_kj` > `ea_ox_kj`.
  Produce: `base_life_days` > `co2_max_pct` > `resp_ref_mgco2_kg_h` > `d_o2_air` (coupled with `film_hole_cm`: one D/L quantity) > `resp_ref_temp_c` > `q10_decay` > `ideal_temp_c`.
* **Part B**: options B1 to B5 in `sim/physics.py` (section 4.4). New empty reference tables with `doi` and `trust` columns: `sorption_params.csv`,
  `oil_stability.csv`, `rh_permeability.csv`, `parameter_priors.csv`. Master gained empty `oil_type` / `oil_type_doi` (section 10).
* **Part C**: `reference/literature_pairings.csv` (8 rows supplied by Adi, all `verified_by_human = False`, no DOI) and `packr/sim/calibrate.py`:
  unknown inputs (RH, fill mass, pack size, temperature range) are swept and reported as intervals; verdict consistent / inconsistent only when the
  whole interval agrees; leave-one-pairing-out fit of at most 3 of the most sensitive unsourced parameters, grid search inside `parameter_priors.csv`
  ranges that carry a DOI, refused with fewer than 3 verified pairings. Report: `reports/calibration.md`.
* v1 on the provisional rows: 1 consistent (cookie MetPP), 2 inconsistent (cookie PP and PVdC/PP: texture never lost at 50 % RH with `crit_aw` 0.55),
  5 undetermined (chips: RH unknown, and v1 has no PV quantity). Median |log error| 0.71. B1 to B3 equal v1 (tables empty, fallback recorded); B4 moves little;
  B5 not applicable. **No official score and no fit** (0 verified pairings, 0 sourced priors). T3's done-when is not met.

**Literature needed, in sensitivity order** (each row needs a DOI):
1. Snacks: GAB `m0, C, K` (or isotherm slope) per snack, Banana Chips first; `crit_aw` with its basis (sensory crispness, not a_w guess); per-snack `oil_type`
   and per-oil PV rate or induction time at 25 C, Ea, km; `qcrit_ml_per_g_fat` or PV end point; headspace volume per g; Ea of WVTR and of oxidation.
2. Produce: `base_life_days`; `co2_max_pct`; respiration rate and its reference temperature; perforation conductance (D/L); `q10_decay`; `ideal_temp_c`;
   temperature fluctuation sd; `ma_gain`.
3. Pairings: verification, DOI, RH and fill mass for the 8 rows (resolve the chips 15-20 C vs 30-35 C anomaly); RH-OTR curves for EVOH, PA and paper.

### 2026-10-26 (T3, literature round): Manikantan et al. 2012 banana chips, continuous moisture score

No simulator parameter, trust value, dataset, split or model changed. 214 tests pass (7 new).

* `reference/literature_pairings.csv`: 15 rows appended (BC-A..BC-N, BC-CTRL) from `data/raw/literature/banana_chips_pairings.csv` (raw file only read;
  sha256 unchanged). `verified_by_human` set to False (the raw file says Yes); `trust = tentative`; "same" expanded. New columns for all rows:
  `initial_moisture_db_pct`, `initial_aw`, `initial_pv_meq_kg`, `film_test_temp_c` (blank = unknown, swept 23 / 30.5 / 38 C), `obs_moisture_db_pct`,
  `obs_aw` (number or ">x"). Observed values filled only where the paper's text confirms them (N, M, D, B, C, control); the raw file's
  automated-reader columns (`moisture_db_pct_d120`, `aw_d120`, their sd, PV, OA) are not copied.
* `reference/sorption_points.csv` (2 banana-chip check points) and `reference/critical_aw_evidence.csv` (Jakubczyk et al. 2008 crispbread, weak,
  no DOI supplied). Neither feeds `sorption_params.csv` or any `crit_aw`.
* `sim/physics.py`: `EXTRA_COLS` gained `moisture_db_at_target_p50`, `aw_at_target_p50` (output only, no random draws added); new `aw_at_time_gab`.
* `sim/calibrate.py`: continuous score (`score_continuous`, `wvtr38_equivalent`, `isotherm_checks`); the pass/fail score skips rows without a
  pass/fail end point. WVTR test RH is held at 90 % (not swept).
* v1 result: nominal day-120 moisture N 4.0 % vs 5.3 % observed, M 5.0 % vs 9.3 %, confirming Adi's estimate. Over the sweep, M is under-predicted by
  at least 2.9 pp (v1 cannot exceed iso_b x RH = 6.4 % d.b.); N, D within their wide intervals; B, C, control (aw > 0.60) undetermined.
  v1's isotherm puts 2.6 % d.b. at aw 0.29 (measured 0.16) and 5.3 % at aw 0.59 (measured 0.48): the line is too steep at low moisture and cannot
  reach 9.3 % at ambient RH, i.e. the measured points are not on any straight line through the origin (slopes 0.16 and 0.11).

### 2026-10-26 (T3, literature round 2): all 15 banana-chip films, empirical isotherm B1e

No change to `commodities_master.csv`, `crit_aw`, `Constants` values, datasets or models. 222 tests pass (8 new; 1 existing data test updated because
the table now holds all 15 films' observations). The raw file was not edited (sha256 unchanged).

* Adi: the day-120 values were NOT checked against the PDF, so `verified_by_human` stays False; the raw file's `Yes` is left as is (data/raw is read-only)
  and overridden in the reference table. New columns `obs_moisture_sd`, `obs_aw_sd`, `obs_pv_meq_kg`, `obs_overall_acceptability`. The text-confirmed
  values of the previous round agree with the reader's table values (asserted while copying).
* `sim/sorption_fit.py`: effective-variance weighted GAB fit (both SDs), bootstrap over films with the initial point kept, identifiability rule
  (bound hits > 5 %, 95 % range > factor 2, |corr| > 0.9). Banana Chips: m0 0.0341 (0.030 to 0.073), C 8.79 (2.8 to 13.1), K 0.923 (0.61 to 1.00,
  on the bound in 34 % of resamples), corr(log m0, log K) -0.95: **unidentifiable**; stored with trust `empirical_fit_constants_unidentifiable`,
  valid aw 0.16 to 0.70. aw at FDC moisture 4.3 % w.b.: 0.40 (0.36 to 0.42); an unweighted fit gives 0.34.
* `physics.py`: `isotherm="gab_empirical"`; `"gab"` now uses literature rows only; valid-range fallback to v1 per trial, recorded in `sim_model`;
  optional row columns `gab_aw_min`, `gab_aw_max`.
* `calibrate.py`: leave-one-film-out (`loo_films`), in-sample B1e reference line, report section. Result (within-paper consistency, not validation):
  moisture within interval v1 6/15 vs B1e-LOO 11/15; nominal RMSE 2.99 vs 2.45 pp; but nominal aw RMSE 0.116 vs 0.161, both biased low, which
  points to too little vapour transport at the nominal conditions (WVTR test conditions / RH), not only the isotherm.
  Film M stays under-predicted (>= 1.5 pp, was >= 2.9); at its measured aw the fold curve gives 6.8 % vs 9.3 % observed.

### 2026-10-26 (T3, round closed)

* **Approved (Adi):** `oil_type = coconut` for Banana Chips (`commodities.OIL_TYPE`), doi 10.1007/s13197-012-0839-0 (Manikantan et al. 2012: chips
  fried in coconut oil). `commodities_master.csv` rebuilt: only `oil_type` and `oil_type_doi` of Banana Chips changed. `oil_stability.csv` stays
  empty until sourced, so B2 still falls back to v1 oxidation for Banana Chips. Datasets and models not regenerated (the master sha256 differs from
  `dataset_summary.json` only in the oil columns, which v1 does not read).
* **B1e stays opt-in.** `Constants().isotherm` remains `"linear"`; dataset generation must not use `isotherm="gab_empirical"` until the 15 Manikantan
  films have `verified_by_human = True`.
* **Remaining moisture under-prediction** (B1e leave-one-film-out nominal bias about -2.3 pp d.b., with nominal aw also biased low) is attributed to
  the unknown film WVTR test conditions (temperature and RH), pending the paper's methods section. If the methods give the test conditions, set
  `film_test_temp_c` (and the test RH) in `literature_pairings.csv` and re-run `python -m packr.sim.calibrate`.
* T3 done-when (direction of >= 3 independent verified pairings) is **not met**: 0 of 23 pairings are verified.

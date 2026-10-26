"""Phase 6: optimisation layer (spec section 8).

    score = w_s * P_success + w_c * (1 - cost_norm) + w_g * sustain + w_m * manufacturability     (weights 0.6 / 0.15 / 0.15 / 0.10)

1. Candidates: every rule-engine structure passing the hard filters (seal and strength minimums from the rule engine). The
   breathable preference for very high respiration is not a filter; the simulator already judges gas exchange.
2. Search variables, inside the sampling bounds of spec 3.4: bulk-layer gauge scale, pack area per kg (unless the caller fixes it),
   and for produce a perforation area designed for a range of in-pack O2 levels, for snacks air or a nitrogen flush.
3. Every variant is scored with the ML surrogate (or, with no model, a short simulator run). Each structure keeps its best variant.
4. The best SHORTLIST structures are re-run through the simulator with VALIDATION_TRIALS trials. That simulated success rate is
   the one reported and the one the final ranking and the "recommended" rule use (spec 7.4).
5. A package with simulated success < 0.5 is never "recommended"; the best available is returned with a note.

Every success rate here is a simulation-based estimate from an uncalibrated simulator with placeholder material values.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .engine.models import Scenario
from .engine.risks import assess_risks
from .engine.rules import Requirements, derive_requirements
from .materials import load_materials
from .profile import Profile
from .sim.dataset import candidate_table, perforation_cm2_for_o2, scaled
from .sim.physics import OUT_COLS, simulate

DEFAULT_WEIGHTS = dict(success=0.6, cost=0.15, sustain=0.15, manufacturability=0.10)
MIN_RECOMMENDED_SUCCESS = 0.5
GAUGE_GRID = (0.5, 0.71, 1.0, 1.41, 2.0)                     # bulk-layer gauge scale, bounds of spec 3.4
AREA_BOUNDS = {"dry_snack": (0.6, 2.5), "fresh_produce": (0.25, 3.0)}     # m2 per kg, spec 3.4
DEFAULT_MASS_KG = {"dry_snack": 0.1, "fresh_produce": 0.5}
N_AREA = 3
PERF_O2_TARGETS = (0.06, 0.10, 0.15, 0.19)                   # in-pack O2 fractions the perforation is designed for (plus no perforation)
FLUSH_O2_PCT = (20.9, 2.0)                                   # snacks: air, or nitrogen flush to about 2 % O2
SHORTLIST = 10
VALIDATION_TRIALS = 1000
SEARCH_TRIALS = 64                                           # per variant, only when no ML model is supplied
TRAINING_RANGE = dict(temp_c=(0.0, 38.0), target_days={"dry_snack": (30, 400), "fresh_produce": (2, 365)},
                      rh_pct={"dry_snack": (30, 90), "fresh_produce": (50, 98)})
DISCLAIMER = "simulation-based estimate; not validated against measured spoilage data"


@dataclass(frozen=True)
class Option:
    rank: int
    layers: str
    n_layers: int
    gauge_scale: float
    thickness_um: float
    otr23: float
    wvtr38: float
    light_barrier: float
    seal: float
    strength: float
    recyclable: bool
    breathable: bool
    cost_rel_per_m2: float
    mass_kg: float
    area_m2_per_kg: float
    perf_mm2_per_kg: float
    flush_o2_pct: float
    map_advice: str
    success_rate: float                 # from the VALIDATION_TRIALS simulator run
    fail_rate: float
    shelf_life_p10: float
    shelf_life_p50: float
    shelf_life_p90: float
    failure_modes: dict                 # p_fail_* from the simulator
    in_pack_o2_pct: float | None
    in_pack_co2_pct: float | None
    ml_success_rate: float | None       # surrogate prediction used for the search (None if no model)
    score: float
    score_parts: dict
    recommended: bool
    success_rate_range: tuple[float, float] | None = None     # across neighbour profiles, when the profile is estimated


@dataclass(frozen=True)
class OptimizeResult:
    options: tuple[Option, ...]
    weights: dict
    requirements: Requirements
    profile: Profile
    n_structures: int
    n_after_filters: int
    n_variants: int
    ranked_by: str
    notes: tuple[str, ...]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return _clean(asdict(self))


def _clean(value):
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def normalise_weights(weights: dict | None) -> dict:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weights {sorted(unknown)}; allowed: {sorted(DEFAULT_WEIGHTS)}")
        w.update({k: float(v) for k, v in weights.items() if v is not None})
    if any(v < 0 or not math.isfinite(v) for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def resp_at_storage(s: Scenario, profile: Profile) -> tuple[float, float]:
    """(respiration at the storage temperature, reference respiration at resp_ref_temp_c), consistent with each other."""
    v = profile.values
    if profile.category == "dry_snack":
        return 0.0, 0.0
    factor = v["q10_resp"] ** ((s.temp_c - v["resp_ref_temp_c"]) / 10.0)
    if s.respiration_mgco2_kg_h is not None:
        return float(s.respiration_mgco2_kg_h), float(s.respiration_mgco2_kg_h) / factor
    return float(v["resp_ref_mgco2_kg_h"]) * factor, float(v["resp_ref_mgco2_kg_h"])


def build_rows(s: Scenario, profile: Profile, packages: pd.DataFrame) -> pd.DataFrame:
    """One simulator / model row per package. `packages` holds the package-spec columns (otr23, wvtr38, ..., flush_o2_pct)."""
    rows = packages.reset_index(drop=True).copy()
    r_t, r_ref = resp_at_storage(s, profile)
    for c, val in profile.values.items():
        rows[c] = val
    rows["resp_ref_mgco2_kg_h"] = r_ref
    rows["chilling_sensitive"] = float(profile.values["chilling_sensitive"])
    rows["category"] = profile.category
    rows["moisture_pct"], rows["fat_pct"], rows["ph"] = s.moisture_pct, s.fat_pct, s.ph
    rows["resp_rate_mgco2_kg_h"] = r_t
    rows["target_days"], rows["temp_c"], rows["rh_pct"] = s.shelf_life_days, s.temp_c, s.rh_pct
    rows["storage"], rows["transport"], rows["harsh"] = s.storage, s.transport, int(s.transport == "Harsh")
    return rows


def search_space(s: Scenario, profile: Profile, req: Requirements, cands: pd.DataFrame, mass_kg: float | None,
                 area_m2_per_kg: float | None) -> pd.DataFrame:
    keep = cands[(cands["seal"] >= req.seal_min) & (cands["strength"] >= req.strength_min)].reset_index(drop=True)
    if keep.empty:
        return keep
    cat = profile.category
    mass = DEFAULT_MASS_KG[cat] if mass_kg is None else mass_kg
    areas = [area_m2_per_kg] if area_m2_per_kg is not None else list(np.geomspace(*AREA_BOUNDS[cat], N_AREA))
    idx = np.repeat(np.arange(len(keep)), len(GAUGE_GRID))
    gs = np.tile(np.array(GAUGE_GRID), len(keep))
    base = pd.concat([keep.iloc[idx].reset_index(drop=True), scaled(keep, idx, gs)], axis=1)
    base["gauge_scale"] = gs
    base["cost_rel_per_m2"] = gs * base["cost_bulk"] + base["cost_fixed"]
    frames = []
    for a in areas:
        if cat == "dry_snack":
            for f in FLUSH_O2_PCT:
                frames.append(base.assign(area_m2_per_kg=a, perf_mm2_per_kg=0.0, flush_o2_pct=f, design_o2=np.nan))
        else:
            r_t, _ = resp_at_storage(s, profile)
            frames.append(base.assign(area_m2_per_kg=a, perf_mm2_per_kg=0.0, flush_o2_pct=20.9, design_o2=np.nan))
            for x in PERF_O2_TARGETS:
                cm2 = perforation_cm2_for_o2(base["otr23"].to_numpy(), a, mass, s.temp_c, r_t, x)
                v = base.assign(area_m2_per_kg=a, perf_mm2_per_kg=cm2 * 100.0 / mass, flush_o2_pct=20.9, design_o2=x)
                frames.append(v[v["perf_mm2_per_kg"] > 0])      # film alone already lets in enough O2: same as no perforation
    out = pd.concat(frames, ignore_index=True)
    out["mass_kg"] = mass
    return out.rename(columns={"light": "light_barrier"})


def score(p_success: np.ndarray, pk: pd.DataFrame, w: dict, cost_max: float) -> tuple[np.ndarray, dict]:
    parts = dict(success=np.asarray(p_success, dtype=float),
                 cost=1.0 - pk["cost_rel_per_m2"].to_numpy() / cost_max,
                 sustain=pk["recyclable"].astype(float).to_numpy(),
                 manufacturability=(pk["seal"].to_numpy() / 5.0 + pk["strength"].to_numpy() / 5.0) / 2.0)
    return sum(w[k] * parts[k] for k in w), parts


def rank_order(p_success: np.ndarray, final_score: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Packages that meet MIN_RECOMMENDED_SUCCESS come first, by score. The rest follow by success rate ("best available"),
    so a cheap recyclable film that fails the target never outranks one that nearly meets it."""
    ok = p_success >= MIN_RECOMMENDED_SUCCESS
    key = np.where(ok, final_score, p_success)
    return np.lexsort((-tiebreak, -key, ~ok))


def _map_advice(row: pd.Series, profile: Profile, req: Requirements) -> str:
    if profile.category == "fresh_produce":
        if row["perf_mm2_per_kg"] > 0:
            return (f"Passive MAP through micro-perforation: about {row['perf_mm2_per_kg']:.3g} mm2 open area per kg, designed for "
                    f"roughly {100 * row['design_o2']:.0f}% O2 at steady state (verify per commodity)")
        return "No perforation: gas exchange through the film alone"
    if row["flush_o2_pct"] < 20.9:
        return f"Nitrogen flush to about {row['flush_o2_pct']:g}% residual O2"
    return "Packed in air (no flush)" + (f"; rule engine says: {req.map_advice}" if req.map_advice != "Not required" else "")


def _range_notes(s: Scenario, cat: str) -> list[str]:
    notes = []
    lo, hi = TRAINING_RANGE["temp_c"]
    if not lo <= s.temp_c <= hi:
        notes.append(f"Temperature {s.temp_c:g} C is outside the simulated range {lo:g} to {hi:g} C; results are extrapolated.")
    for key, val in (("target_days", s.shelf_life_days), ("rh_pct", s.rh_pct)):
        lo, hi = TRAINING_RANGE[key][cat]
        if not lo <= val <= hi:
            notes.append(f"{key} {val:g} is outside the range sampled for {cat} ({lo:g} to {hi:g}); the ML ranking is extrapolating.")
    return notes


def optimize(s: Scenario, profile: Profile, model: dict | None = None, weights: dict | None = None, top_k: int = 3,
             materials: pd.DataFrame | None = None, mass_kg: float | None = None, area_m2_per_kg: float | None = None,
             neighbour_profiles: list[Profile] | None = None, seed: int = 0) -> OptimizeResult:
    if s.storage == "Frozen":
        raise ValueError("Frozen storage is not modelled by the simulator; no success rate can be estimated")
    w = normalise_weights(weights)
    materials = load_materials() if materials is None else materials
    req = derive_requirements(s, assess_risks(s))
    cands = candidate_table(materials)
    space = search_space(s, profile, req, cands, mass_kg, area_m2_per_kg)
    notes = list(profile.notes) + _range_notes(s, profile.category)
    if (materials["trust"] == "placeholder").any():
        notes.append("Material barrier and cost values are placeholders, not datasheet or literature values.")
    if space.empty:
        notes.append("No structure meets the rule engine's seal and strength minimums.")
        return OptimizeResult((), w, req, profile, len(cands), 0, 0, "none", tuple(notes))

    rows = build_rows(s, profile, space)
    if model is not None:
        from .ml.predict import predict
        p_search = predict(model, rows)["success_rate"].to_numpy()
        ranked_by = "ml_surrogate"
    else:
        p_search = simulate(rows, n_trials=SEARCH_TRIALS, seed=seed)["success_rate"].to_numpy(dtype=float)
        ranked_by = f"simulator_{SEARCH_TRIALS}_trials"
    cost_max = float(space["cost_rel_per_m2"].max())
    sc, _ = score(p_search, space, w, cost_max)
    space = space.assign(search_p=p_search, search_score=sc)
    best = space.loc[space.groupby("layers")["search_score"].idxmax()].sort_values("search_score", ascending=False)
    short = best.head(SHORTLIST).reset_index(drop=True)

    sim = simulate(build_rows(s, profile, short), n_trials=VALIDATION_TRIALS, seed=seed + 1).reset_index(drop=True)
    p_sim = sim["success_rate"].to_numpy(dtype=float)
    final, parts = score(p_sim, short, w, cost_max)
    order = rank_order(p_sim, final, short["search_score"].to_numpy())[:top_k]

    ranges = {}
    if neighbour_profiles:
        picked = short.iloc[order].reset_index(drop=True)
        rs = [simulate(build_rows(s, p, picked), n_trials=VALIDATION_TRIALS // 4, seed=seed + 2)["success_rate"].to_numpy(dtype=float)
              for p in neighbour_profiles]
        allp = np.vstack(rs + [sim["success_rate"].to_numpy(dtype=float)[order]])
        ranges = {i: (float(allp[:, j].min()), float(allp[:, j].max())) for j, i in enumerate(order)}

    options = []
    for rank, i in enumerate(order, 1):
        r, m = short.iloc[i], sim.iloc[i]
        ps = float(m["success_rate"])
        options.append(Option(
            rank=rank, layers=r["layers"], n_layers=int(r["n_layers"]), gauge_scale=float(r["gauge_scale"]), thickness_um=float(r["thickness_um"]),
            otr23=float(r["otr23"]), wvtr38=float(r["wvtr38"]), light_barrier=float(r["light_barrier"]), seal=float(r["seal"]),
            strength=float(r["strength"]), recyclable=bool(r["recyclable"]), breathable=bool(r["breathable"]),
            cost_rel_per_m2=float(r["cost_rel_per_m2"]), mass_kg=float(r["mass_kg"]), area_m2_per_kg=float(r["area_m2_per_kg"]),
            perf_mm2_per_kg=float(r["perf_mm2_per_kg"]), flush_o2_pct=float(r["flush_o2_pct"]), map_advice=_map_advice(r, profile, req),
            success_rate=ps, fail_rate=1.0 - ps, shelf_life_p10=float(m["shelf_life_p10"]), shelf_life_p50=float(m["shelf_life_p50"]),
            shelf_life_p90=float(m["shelf_life_p90"]),
            failure_modes={k: float(m[k]) for k in OUT_COLS if k.startswith("p_fail_")},
            in_pack_o2_pct=None if profile.category == "dry_snack" else float(m["x_o2_pct_p50"]),
            in_pack_co2_pct=None if profile.category == "dry_snack" else float(m["x_co2_pct_p50"]),
            ml_success_rate=float(r["search_p"]) if model is not None else None,
            score=float(final[i]), score_parts={k: float(v[i]) for k, v in parts.items()},
            recommended=ps >= MIN_RECOMMENDED_SUCCESS, success_rate_range=ranges.get(i)))

    if not any(o.recommended for o in options):
        notes.append(f"No package meets the target: the best simulated success rate is {max(o.success_rate for o in options):.2f} "
                     f"(< {MIN_RECOMMENDED_SUCCESS}). Shown for comparison only; consider a shorter target, colder storage or another format.")
    if req.prefer_breathable and options and not options[0].breathable and options[0].perf_mm2_per_kg == 0:
        notes.append("The rule engine prefers a breathable film for this respiration rate; the top option is neither breathable nor perforated.")
    if s.respiration_mgco2_kg_h is None and profile.category == "fresh_produce":
        notes.append("Respiration taken from the commodity profile and Q10-scaled to the storage temperature.")
    return OptimizeResult(tuple(options), w, req, profile, len(cands), int(space["layers"].nunique()), len(space), ranked_by, tuple(notes))

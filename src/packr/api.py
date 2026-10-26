"""Phase 7: HTTP API (spec section 9.1).

    uvicorn packr.api:app --reload             # needs the api extra: pip install -e ".[api]"

POST /recommend   a Scenario (validated by `packr.engine.Scenario`) plus optional `commodity_profile` overrides, `weights`,
                  `category` (for a food not in the table) and pack geometry. Returns the top-3 packages, each with a simulated
                  success rate (1,000 trials), shelf-life percentiles, failure-mode breakdown and MAP advice, plus the rule trace,
                  `profile_source` and the disclaimer.
GET  /commodities the commodity table.

The ML surrogate (models/packr_gbt.joblib, or PACKR_MODEL) ranks candidates; if no model is found the candidates are ranked by a
short simulator run instead and the response says so. Every success rate is a simulation-based estimate.
"""
from __future__ import annotations

import math
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from .commodities import load_commodities
from .engine import Scenario, recommend
from .engine.risks import assess_risks
from .engine.rules import derive_requirements
from .optimize import DEFAULT_WEIGHTS, DISCLAIMER, optimize
from .profile import PROFILE_COLS, resolve_profile

ESTIMATE_LABEL = "simulation-based estimate"
_Q10_DEFAULT, _REF_T = 2.5, 10.0        # only used to place an unknown food's respiration among its neighbours
SHELF_LIFE_CAP_DAYS = 5000.0             # the simulator reports "no failure" as this value

ProfileOverrides = create_model("ProfileOverrides", __config__=ConfigDict(extra="forbid"),
                                **{c: (float | None, None) for c in PROFILE_COLS})


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: float | None = Field(None, ge=0)
    cost: float | None = Field(None, ge=0)
    sustain: float | None = Field(None, ge=0)
    manufacturability: float | None = Field(None, ge=0)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commodity: str = Field(min_length=1)
    moisture_pct: float
    fat_pct: float
    ph: float
    shelf_life_days: float
    temp_c: float
    rh_pct: float
    storage: str = "Ambient"
    transport: str = "Normal"
    respiration: str = "None"
    respiration_mgco2_kg_h: float | None = None
    category: Literal["fresh_produce", "dry_snack"] | None = None
    commodity_profile: ProfileOverrides | None = None
    weights: Weights | None = None
    mass_kg: float | None = Field(None, gt=0, le=50)
    area_m2_per_kg: float | None = Field(None, gt=0, le=20)

    def scenario(self) -> Scenario:
        return Scenario(self.commodity, self.moisture_pct, self.fat_pct, self.ph, self.shelf_life_days, self.temp_c, self.rh_pct,
                        storage=self.storage, transport=self.transport, respiration=self.respiration,
                        respiration_mgco2_kg_h=self.respiration_mgco2_kg_h)


def _finite(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _package(o) -> dict:
    return dict(
        rank=o.rank, layers=o.layers.split("/"), n_layers=o.n_layers, thickness_um=round(o.thickness_um, 1), gauge_scale=round(o.gauge_scale, 2),
        otr_cm3_m2_day=o.otr23, wvtr_g_m2_day=o.wvtr38, light_barrier=o.light_barrier, seal=o.seal, strength=round(o.strength, 2),
        recyclable=o.recyclable, breathable=o.breathable, cost_rel_per_m2=round(o.cost_rel_per_m2, 3),
        pack=dict(mass_kg=o.mass_kg, area_m2_per_kg=round(o.area_m2_per_kg, 3), perforation_mm2_per_kg=o.perf_mm2_per_kg,
                  flush_o2_pct=o.flush_o2_pct),
        map_advice=o.map_advice, success_rate=o.success_rate, fail_rate=o.fail_rate, success_rate_label=ESTIMATE_LABEL,
        success_rate_range=list(o.success_rate_range) if o.success_rate_range else None,
        shelf_life_p10=_finite(o.shelf_life_p10), shelf_life_p50=_finite(o.shelf_life_p50), shelf_life_p90=_finite(o.shelf_life_p90),
        failure_modes=o.failure_modes, in_pack_o2_pct=o.in_pack_o2_pct, in_pack_co2_pct=o.in_pack_co2_pct,
        ml_success_rate=o.ml_success_rate, score=round(o.score, 4), score_parts={k: round(v, 4) for k, v in o.score_parts.items()},
        recommended=o.recommended)


def _profile_block(p) -> dict:
    return dict(commodity=p.commodity, category=p.category, in_table=p.in_table, neighbours=list(p.neighbours),
                estimated_fields=list(p.estimated_fields), user_fields=list(p.user_fields),
                values={k: _finite(v) for k, v in p.values.items()})


def _frozen_response(s: Scenario) -> dict:
    """Frozen storage is accepted but not modelled (spec ground rule 5): rule-engine structures only, no success rate."""
    rec = recommend(s)
    return dict(
        supported=False, disclaimer=DISCLAIMER, profile_source=None, profile=None,
        packages=[dict(rank=i, layers=list(st.layers), n_layers=st.n, thickness_um=st.thickness_um, otr_cm3_m2_day=st.OTR,
                       wvtr_g_m2_day=st.WVTR, light_barrier=st.light, seal=st.seal, strength=st.strength, recyclable=st.recyclable,
                       map_advice=rec.requirements.map_advice, success_rate=None, fail_rate=None, recommended=False)
                  for i, st in enumerate(rec.top, 1)],
        success_rate=None, fail_rate=None, success_rate_label=ESTIMATE_LABEL, shelf_life_p10=None, shelf_life_p50=None, shelf_life_p90=None,
        failure_modes=None, recommended=False, rule_trace=list(rec.risks.reasons) + list(rec.requirements.trace),
        weights=None, ranked_by="rule_engine", model=None,
        notes=["Frozen storage is not modelled: no success rate or shelf life can be estimated. Structures shown pass the rule engine only.",
               *rec.notes])


def create_app(model: dict | None = None, load_default_model: bool = True) -> FastAPI:
    app = FastAPI(title="Packr", version="0.1.0",
                  description="Food packaging recommendation. Every success rate is a simulation-based estimate from an uncalibrated "
                              "simulator; it is not validated against measured spoilage data.")
    state = {"model": model, "model_error": None}
    if model is None and load_default_model:
        try:
            from .ml.predict import load_model
            state["model"] = load_model()
        except (FileNotFoundError, ImportError) as e:
            state["model_error"] = str(e)

    @app.get("/commodities")
    def commodities() -> dict:
        cm = load_commodities()
        cols = ["commodity_id", "commodity", "category", "subgroup", "moisture_pct", "fat_pct", "ph", "resp_ref_mgco2_kg_h", "ideal_temp_c",
                "ideal_rh_pct", "base_life_days", "trust_composition", "trust_respiration", "trust_sim_params"]
        rows = cm[cols].astype(object).where(cm[cols].notna(), None).to_dict("records")
        return dict(count=len(rows), commodities=rows)

    @app.post("/recommend")
    def recommend_endpoint(req: RecommendRequest) -> dict:
        try:
            s = req.scenario()
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        if s.storage == "Frozen":
            return _frozen_response(s)
        overrides = req.commodity_profile.model_dump(exclude_none=True) if req.commodity_profile else {}
        resp_ref = (s.respiration_mgco2_kg_h / _Q10_DEFAULT ** ((s.temp_c - _REF_T) / 10.0)
                    if s.respiration_mgco2_kg_h is not None else None)
        try:
            profile = resolve_profile(s.commodity, s.moisture_pct, s.fat_pct, category=req.category, overrides=overrides, resp_ref=resp_ref)
            neighbours = [resolve_profile(n, s.moisture_pct, s.fat_pct, overrides=overrides) for n in profile.neighbours] \
                if profile.source == "estimated" else None
            res = optimize(s, profile, model=state["model"], weights=req.weights.model_dump(exclude_none=True) if req.weights else None,
                           mass_kg=req.mass_kg, area_m2_per_kg=req.area_m2_per_kg, neighbour_profiles=neighbours)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        packages = [_package(o) for o in res.options]
        top = packages[0] if packages else None
        notes = list(res.notes)
        if any(p["shelf_life_p90"] is not None and p["shelf_life_p90"] >= SHELF_LIFE_CAP_DAYS for p in packages):
            notes.append(f"A shelf life of {SHELF_LIFE_CAP_DAYS:g} days is the simulator's cap: no failure was simulated within that horizon.")
        if state["model"] is None:
            notes.append("No trained ML model found; candidates were ranked by a short simulator run instead.")
        risks = assess_risks(s)
        m = state["model"]
        return dict(
            supported=True, disclaimer=DISCLAIMER, profile_source=profile.source, profile=_profile_block(profile),
            packages=packages,
            success_rate=top["success_rate"] if top else None, fail_rate=top["fail_rate"] if top else None, success_rate_label=ESTIMATE_LABEL,
            success_rate_range=top["success_rate_range"] if top else None,
            shelf_life_p10=top["shelf_life_p10"] if top else None, shelf_life_p50=top["shelf_life_p50"] if top else None,
            shelf_life_p90=top["shelf_life_p90"] if top else None, failure_modes=top["failure_modes"] if top else None,
            recommended=bool(top and top["recommended"]),
            rule_trace=list(risks.reasons) + list(derive_requirements(s, risks).trace),
            weights=res.weights, ranked_by=res.ranked_by,
            model=None if m is None else dict(path=m.get("path"), created=m.get("created"), n_commodities=len(m.get("trained_on", []))),
            search=dict(structures=res.n_structures, after_filters=res.n_after_filters, variants=res.n_variants),
            shelf_life_cap_days=SHELF_LIFE_CAP_DAYS,
            notes=notes)

    @app.get("/")
    def root() -> dict:
        return dict(service="packr", endpoints=["POST /recommend", "GET /commodities"], default_weights=DEFAULT_WEIGHTS, disclaimer=DISCLAIMER)

    return app


app = create_app()

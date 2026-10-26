import math

import numpy as np
import pytest

from packr.engine import Scenario
from packr.engine.risks import assess_risks
from packr.engine.rules import derive_requirements
from packr.optimize import DEFAULT_WEIGHTS, GAUGE_GRID, MIN_RECOMMENDED_SUCCESS, normalise_weights, optimize, rank_order, search_space
from packr.materials import load_materials
from packr.profile import resolve_profile
from packr.sim.dataset import candidate_table

CHIPS = Scenario("Potato Chips", 2, 33, 6.0, 180, 25, 60)
CHIPS_HARSH = Scenario("Potato Chips", 2, 33, 6.0, 180, 25, 60, transport="Harsh")
STRAWBERRY = Scenario("Strawberries", 91, 0.3, 3.5, 5, 2, 90, storage="Chilled")


def prof(s, **kw):
    return resolve_profile(s.commodity, s.moisture_pct, s.fat_pct, **kw)


@pytest.fixture(scope="module")
def chips():
    return optimize(CHIPS, prof(CHIPS))


# --- weights and ordering ---------------------------------------------------------------------------------

def test_default_weights_are_the_spec_values_and_normalise():
    assert DEFAULT_WEIGHTS == dict(success=0.6, cost=0.15, sustain=0.15, manufacturability=0.10)
    w = normalise_weights({"success": 3, "cost": 1, "sustain": 0, "manufacturability": 0})
    assert math.isclose(sum(w.values()), 1) and math.isclose(w["success"], 0.75)
    for bad in ({"success": -1}, {"speed": 1}, {k: 0 for k in DEFAULT_WEIGHTS}):
        with pytest.raises(ValueError):
            normalise_weights(bad)


def test_rank_order_puts_packages_meeting_the_target_first():
    p = np.array([0.30, 0.90, 0.60, 0.45])
    score = np.array([0.95, 0.70, 0.80, 0.10])          # the 0.30 package has the best score but fails the target
    order = rank_order(p, score, np.zeros(4))
    assert list(order) == [2, 1, 3, 0]                   # passing by score (0.80, 0.70), then failing by success (0.45, 0.30)


def test_top3_is_ranked_by_score_with_distinct_structures(chips):
    assert len(chips.options) == 3 and chips.disclaimer.startswith("simulation-based estimate")
    assert len({o.layers for o in chips.options}) == 3
    rec = [o for o in chips.options if o.recommended]
    assert [o.score for o in rec] == sorted((o.score for o in rec), reverse=True)
    assert [o.rank for o in chips.options] == [1, 2, 3]
    assert math.isclose(sum(chips.weights.values()), 1)
    for o in chips.options:
        assert o.recommended == (o.success_rate >= MIN_RECOMMENDED_SUCCESS)
        assert math.isclose(o.success_rate + o.fail_rate, 1)
        assert math.isclose(o.score, sum(chips.weights[k] * o.score_parts[k] for k in chips.weights))


def test_success_only_weights_rank_by_simulated_success():
    r = optimize(CHIPS, prof(CHIPS), weights={"success": 1, "cost": 0, "sustain": 0, "manufacturability": 0})
    rates = [o.success_rate for o in r.options if o.recommended]
    assert rates == sorted(rates, reverse=True)


def test_hard_filters_follow_the_rule_engine():
    req = derive_requirements(CHIPS_HARSH, assess_risks(CHIPS_HARSH))
    r = optimize(CHIPS_HARSH, prof(CHIPS_HARSH))
    assert r.n_after_filters < r.n_structures
    for o in r.options:
        assert o.seal >= req.seal_min and o.strength >= req.strength_min


def test_cost_per_m2_scales_with_gauge_for_bulk_layers_only():
    s, p = CHIPS, prof(CHIPS)
    req = derive_requirements(s, assess_risks(s))
    space = search_space(s, p, req, candidate_table(load_materials()), None, 1.0)
    bulk = space[space["layers"] == "BOPP/CPP"].drop_duplicates("gauge_scale").sort_values("gauge_scale")
    assert list(bulk["gauge_scale"]) == list(GAUGE_GRID)
    assert bulk["cost_rel_per_m2"].is_monotonic_increasing and np.isclose(bulk["cost_rel_per_m2"].iloc[2], 2.0)   # BOPP 1 + CPP 1
    assert bulk["otr23"].is_monotonic_decreasing


def test_produce_search_includes_designed_perforations_and_reports_gas():
    r = optimize(STRAWBERRY, prof(STRAWBERRY))
    assert r.profile.commodity == "Strawberry" and r.profile.source == "table"
    assert all(o.in_pack_o2_pct is not None for o in r.options)
    assert any(o.perf_mm2_per_kg > 0 for o in r.options)


def test_no_package_meeting_the_target_is_never_recommended():
    hopeless = Scenario("Strawberry", 91, 0.3, 3.5, 60, 25, 70)
    r = optimize(hopeless, prof(hopeless))
    assert not any(o.recommended for o in r.options)
    assert any("No package meets the target" in n for n in r.notes)


def test_frozen_storage_is_refused():
    with pytest.raises(ValueError, match="Frozen"):
        optimize(Scenario("Potato Chips", 2, 33, 6.0, 180, -18, 60, storage="Frozen"), prof(CHIPS))


def test_estimated_profile_reports_a_range():
    s = Scenario("Mystery crisps", 3, 28, 6.0, 120, 25, 60)
    p = prof(s)
    nb = [resolve_profile(n, s.moisture_pct, s.fat_pct) for n in p.neighbours]
    r = optimize(s, p, neighbour_profiles=nb)
    assert p.source == "estimated" and len(p.neighbours) == 3
    for o in r.options:
        lo, hi = o.success_rate_range
        assert lo <= o.success_rate <= hi

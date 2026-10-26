import json
import math

import pytest

from packr.engine import Scenario, format_report, recommend
from packr.engine.models import respiration_level_from_rate
from packr.engine.structures import build_candidates, check, evaluate
from packr.engine.risks import assess_risks
from packr.engine.rules import derive_requirements
from packr.materials import load_materials

CHIPS = Scenario("Potato Chips", 2, 33, 6.0, 180, 25, 60)
STRAWBERRY = Scenario("Strawberries", 91, 0.3, 3.5, 7, 5, 90, storage="Chilled", respiration="High")
HOT_HARSH = Scenario("My product", 4, 22, 6.2, 120, 32, 75, transport="Harsh")


def layers(rec):
    return [st.layers for st in rec.top]


# --- regression against the demo notebook's results --------------------------------------------------

def test_chips_matches_demo():
    rec = recommend(CHIPS)
    assert rec.n_scanned == 89 and rec.n_passing == 25
    assert layers(rec) == [("MetPET", "CPP"), ("MetPET", "LDPE"), ("BOPP", "AlFoil", "CPP")]
    best = rec.top[0]
    assert best.OTR == pytest.approx(1.499, abs=1e-3)
    assert best.WVTR == pytest.approx(0.889, abs=1e-3)
    assert best.thickness_um == 37 and best.recyclable is False
    assert rec.requirements.otr_max == 5 and rec.requirements.wvtr_max == 1
    assert "nitrogen" in rec.requirements.map_advice


def test_strawberry_flips_to_breathable_film():
    rec = recommend(STRAWBERRY)
    assert rec.n_passing == 2
    assert layers(rec) == [("MicroperfBOPP",), ("CPP",)]
    assert rec.requirements.otr_min == 2000 and math.isinf(rec.requirements.otr_max)
    assert rec.requirements.prefer_breathable and rec.top[0].breathable
    assert "passive MAP" in rec.requirements.map_advice


def test_hot_harsh_scenario_matches_demo():
    rec = recommend(HOT_HARSH)
    assert rec.n_passing == 8
    assert layers(rec) == [("MetPET", "CPP"), ("BOPP", "MetPET", "CPP"), ("MetPET", "MetBOPP", "CPP")]
    assert rec.risks.levels["oxidation"] == 3                     # 22 % fat -> High, already at the cap
    assert rec.requirements.strength_min == 4


def test_rejections_are_explained():
    mats = load_materials()
    req = derive_requirements(CHIPS, assess_risks(CHIPS))
    reasons = check(evaluate(("LDPE",), mats), req)
    assert any("OTR" in r for r in reasons) and any("light barrier" in r for r in reasons)
    assert check(evaluate(("MetPET", "CPP"), mats), req) == []


# --- behaviour ---------------------------------------------------------------------------------------

def test_candidate_count_and_unique():
    cands = build_candidates(load_materials())
    assert len(cands) == 89 and len(set(cands)) == 89
    assert all(c[-1] in {"LDPE", "HDPE", "CPP"} or len(c) == 1 for c in cands)   # laminates end in a sealant


def test_numeric_respiration_overrides_class():
    s = Scenario("x", 90, 0.3, 3.5, 7, 5, 90, respiration="None", respiration_mgco2_kg_h=30)
    assert s.respiration_level == 3
    assert [respiration_level_from_rate(r) for r in (3, 5, 8, 15, 21, 70)] == [0, 0, 1, 2, 3, 3]


def test_warm_storage_raises_oxidation_one_level():
    cool = assess_risks(Scenario("x", 4, 12, 6, 120, 25, 50)).levels["oxidation"]
    warm = assess_risks(Scenario("x", 4, 12, 6, 120, 32, 50)).levels["oxidation"]
    assert (cool, warm) == (2, 3)


def test_short_shelf_life_relaxes_limits():
    short = Scenario("Chips", 2, 33, 6.0, 60, 25, 60)
    req = derive_requirements(short, assess_risks(short))
    assert req.otr_max == 50 and req.wvtr_max == 5


def test_no_solution_is_reported_not_raised():
    impossible = Scenario("x", 2, 33, 6.0, 365, 25, 60, transport="Harsh")
    mats = load_materials()
    mats = mats.drop(index=["AlFoil", "MetPET", "MetBOPP", "EVOH"])    # remove every real barrier
    rec = recommend(impossible, materials=mats)
    assert rec.top == () and rec.n_passing == 0
    assert any("No candidate" in n for n in rec.notes)
    assert "NOTE" in format_report(rec)


def test_notes_flag_placeholders_and_unused_storage_type():
    rec = recommend(Scenario("Frozen peas", 78, 0.4, 6.3, 365, -18, 80, storage="Frozen"))
    joined = " ".join(rec.notes)
    assert "placeholders" in joined and "Frozen" in joined


# --- input validation --------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(moisture_pct=101), dict(moisture_pct=-1), dict(fat_pct=120), dict(moisture_pct=70, fat_pct=40),
    dict(ph=15), dict(shelf_life_days=0), dict(rh_pct=101), dict(temp_c=100),
    dict(storage="Warm"), dict(transport="Rough"), dict(respiration="Extreme"),
    dict(moisture_pct=float("nan")), dict(respiration_mgco2_kg_h=-3),
])
def test_scenario_rejects_nonsense(kwargs):
    base = dict(commodity="x", moisture_pct=10, fat_pct=5, ph=6, shelf_life_days=30, temp_c=20, rh_pct=50)
    with pytest.raises(ValueError):
        Scenario(**{**base, **kwargs})


def test_from_dict_accepts_demo_notebook_keys():
    s = Scenario.from_dict(dict(commodity="Potato Chips", moisture=2, fat=33, ph=6.0, respiration="None",
                                shelf_life_days=180, temp_c=25, rh=60, storage="Ambient", transport="Normal"))
    assert s == CHIPS


# --- output ------------------------------------------------------------------------------------------

def test_recommendation_is_json_serialisable_and_infinity_safe():
    d = recommend(STRAWBERRY).to_dict()
    text = json.dumps(d)                                   # would raise on inf / non-serialisable types
    assert d["requirements"]["otr_max"] is None
    assert d["top"][0]["layers"] == ["MicroperfBOPP"]
    assert json.loads(text)["n_passing"] == 2


def test_report_contains_the_expected_sections():
    text = format_report(recommend(CHIPS))
    for part in ("WHY (Level 1 risks)", "RECOMMENDED PACKAGING", "layer 1 (outer): MetPET",
                 "layer 2 (sealant): CPP", "REQUIRED PROPERTIES", "MAP", "TOP ALTERNATIVES"):
        assert part in text
    assert "At least 2000" in format_report(recommend(STRAWBERRY))

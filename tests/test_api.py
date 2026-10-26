import pytest
from fastapi.testclient import TestClient

from packr.api import ESTIMATE_LABEL, create_app
from packr.optimize import DISCLAIMER

CHIPS = {"commodity": "Potato Chips", "moisture_pct": 2, "fat_pct": 33, "ph": 6.0, "shelf_life_days": 180, "temp_c": 25, "rh_pct": 60}


@pytest.fixture(scope="module")
def client():
    # no ML model: candidates are ranked by a short simulator run, which keeps the test independent of trained files
    return TestClient(create_app(load_default_model=False))


def post(client, **kw):
    return client.post("/recommend", json=CHIPS | kw)


def test_recommend_returns_top3_with_disclaimer_and_profile_source(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER and "not validated" in body["disclaimer"]
    assert body["profile_source"] == "table" and body["supported"] is True
    assert len(body["packages"]) == 3 and body["success_rate_label"] == ESTIMATE_LABEL
    top = body["packages"][0]
    for key in ("layers", "thickness_um", "otr_cm3_m2_day", "wvtr_g_m2_day", "light_barrier", "seal", "strength", "map_advice",
                "success_rate", "fail_rate", "shelf_life_p10", "shelf_life_p50", "shelf_life_p90", "failure_modes"):
        assert key in top
    assert all(p["success_rate_label"] == ESTIMATE_LABEL for p in body["packages"])
    assert body["success_rate"] == top["success_rate"] and abs(body["success_rate"] + body["fail_rate"] - 1) < 1e-9
    assert any("OTR" in t for t in body["rule_trace"])
    assert body["ranked_by"].startswith("simulator") and any("No trained ML model" in n for n in body["notes"])


def test_unknown_food_gets_an_estimated_profile_and_a_range(client):
    body = post(client, commodity="Masala Crisps", fat_pct=30).json()
    assert body["profile_source"] == "estimated" and len(body["profile"]["neighbours"]) == 3
    lo, hi = body["success_rate_range"]
    assert lo <= body["success_rate"] <= hi


def test_user_overrides_set_profile_source_user(client):
    body = post(client, commodity_profile={"crit_aw": 0.35}).json()
    assert body["profile_source"] == "user" and body["profile"]["values"]["crit_aw"] == 0.35


def test_custom_weights_are_normalised_and_echoed(client):
    body = post(client, weights={"success": 1, "cost": 1, "sustain": 0, "manufacturability": 0}).json()
    assert body["weights"] == {"success": 0.5, "cost": 0.5, "sustain": 0.0, "manufacturability": 0.0}


@pytest.mark.parametrize("bad", [
    {"moisture_pct": 120},                       # Scenario validation
    {"moisture_pct": 60, "fat_pct": 50},         # Scenario: moisture + fat > 100
    {"storage": "Hot"},
    {"transport": "Rough"},
    {"colour": "red"},                           # unknown field
    {"commodity_profile": {"bogus": 1}},
    {"weights": {"success": -1}},
    {"weights": {"success": 0, "cost": 0, "sustain": 0, "manufacturability": 0}},
    {"category": "dairy"},
    {"commodity": "Potato Chips", "category": "fresh_produce"},
])
def test_invalid_requests_are_rejected(client, bad):
    r = post(client, **bad)
    assert r.status_code == 422, r.text


def test_frozen_is_accepted_but_not_scored(client):
    body = post(client, storage="Frozen", temp_c=-18).json()
    assert body["supported"] is False and body["success_rate"] is None
    assert body["packages"] and all(p["success_rate"] is None for p in body["packages"])
    assert any("Frozen storage is not modelled" in n for n in body["notes"])


def test_commodities_lists_the_table(client):
    body = client.get("/commodities").json()
    assert body["count"] == 48 and {"Potato Chips", "Strawberry"} <= {c["commodity"] for c in body["commodities"]}

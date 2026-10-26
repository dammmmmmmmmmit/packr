import pandas as pd
import pytest

from packr.materials import MATERIALS_CSV, load_materials, required_thickness_um
from packr.units import otr_from_permeability

# the indicative values the original demo notebook used: (OTR cm3/m2/d, WVTR g/m2/d) at nominal gauge
DEMO = {"LDPE": (6500, 18), "HDPE": (1600, 6), "CPP": (3000, 8), "BOPP": (1600, 6), "PET": (100, 40),
        "BOPA": (40, 250), "Paper": (10000, 500), "EVOH": (1, 120), "AlFoil": (0.01, 0.01),
        "MetPET": (1.5, 1.0), "MetBOPP": (80, 0.5), "MicroperfBOPP": (8000, 12)}


def test_derived_barrier_values_reproduce_the_demo_numbers():
    mats = load_materials()
    assert set(mats.index) == set(DEMO)
    for name, (otr, wvtr) in DEMO.items():
        assert mats.loc[name, "OTR"] == pytest.approx(otr, rel=1e-4), name
        assert mats.loc[name, "WVTR"] == pytest.approx(wvtr, rel=1e-4), name


def test_every_material_is_flagged_as_placeholder():
    assert (load_materials()["trust"] == "placeholder").all()


def test_thickness_scaling_for_bulk_polymers_only():
    mats = load_materials()
    p = mats.loc["LDPE", "p_o2"]
    assert otr_from_permeability(p, 100) == pytest.approx(mats.loc["LDPE", "OTR"] / 2)   # twice as thick, half the OTR
    assert required_thickness_um(mats, "LDPE", otr_max=3250) == pytest.approx(100)
    with pytest.raises(ValueError):
        required_thickness_um(mats, "MetPET", otr_max=1)


def test_loader_rejects_bad_tables(tmp_path):
    good = pd.read_csv(MATERIALS_CSV)
    good.drop(columns=["trust"]).to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_materials(tmp_path / "a.csv")

    dup = pd.concat([good, good.iloc[[0]]])
    dup.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_materials(tmp_path / "b.csv")

    bad_role = good.copy()
    bad_role.loc[0, "roles"] = "sealant;wrapper"
    bad_role.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="unknown roles"):
        load_materials(tmp_path / "c.csv")

    no_barrier = good.copy()
    no_barrier.loc[0, ["p_o2", "p_wv"]] = float("nan")
    no_barrier.to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(ValueError, match="without usable barrier"):
        load_materials(tmp_path / "d.csv")

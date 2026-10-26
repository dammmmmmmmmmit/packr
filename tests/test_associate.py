import numpy as np
import pandas as pd
import pytest

from packr.associate import PLAUSIBLE_RANGE, associate, canonical, compare_with_placeholders, doc_materials
from packr.materials import MATERIALS_CSV
from packr.transmat import build_quantities
from packr.units import GAS_BASE

# packaging mentions copied from data/raw/goldenTRANSMAT-all.csv


@pytest.mark.parametrize("mention, bases", [
    ("PLA / Cloisite 30B composite film", {"PLA"}),                      # filler removed, matrix kept
    ("2C18 ammonium OMMT-PP composites", {"PP (grade not stated)"}),
    ("films plasticized by polyethylene glycol", set()),                 # PEG is an additive, not polyethylene
    ("NY / mLLDPE film", {"PA (nylon)", "LLDPE"}),                       # a laminate maps to both layers
    ("chitosan coated PE", {"chitosan", "PE (grade not stated)"}),
    ("pristine poly ( n-butyl methacrylate ) membrane", {"PBMA"}),
    ("fish gelatin / nanoclay composite films", {"fish gelatin"}),
])
def test_canonical_mentions(mention, bases):
    b, _, leftover = canonical(mention)
    assert b == bases and leftover == ""


def test_ambiguous_words_stay_unmapped():
    assert canonical("polyester–clay nanocomposite")[2] == "polyester"       # PET or PLA? not guessed


def frame(doc, mentions, quantities):
    rows = [dict(id=i, doc=doc, target="packaging", original_value=str(m.split()), attached_value=None, type="SYMBOLIC", annotator=1, doi="10.x/" + doc)
            for i, m in enumerate(mentions)]
    rows += [dict(id=100 + i, doc=doc, target="permeability", original_value=o, attached_value=a, type="QUANTITY", annotator=1, doi="10.x/" + doc)
             for i, (o, a) in enumerate(quantities)]
    return pd.DataFrame(rows)


PP_O2 = ("(['89'], ['cm', '^', '3', 'μm', '/', 'm', '^', '2', 'day', 'mmHg'])", "['oxygen', 'permeation']")


def test_single_material_paper_is_associated_and_flagged_unresolved_variant():
    df = frame("pp", ["pure PP", "OMMT-polypropylene nanocomposites"], [PP_O2])
    mat, unres, docs = associate(df, build_quantities(df))
    assert docs.loc[0, "single_material"] and not docs.loc[0, "single_name"]
    assert len(mat) == 1 and unres.empty
    r = mat.iloc[0]
    assert r["base_material"] == "PP (grade not stated)" and r["association"] == "single-material paper"
    assert r["variant_resolved"] == False and r["doi"] == "10.x/pp"                                    # noqa: E712
    assert r["permeability"] == pytest.approx(89 * 1e-3 * 101325 / 133.322387415)


def test_multi_material_paper_goes_to_unresolved_never_guessed():
    df = frame("mc", ["Methylcellulose films", "low density polyethylene films"],
               [("(['8.7', '×', '10', '^', '−', '13'], ['g', '/', 'm', 's', 'Pa'])", "['WVP']")])
    mat, unres, _ = associate(df, build_quantities(df))
    assert mat.empty and len(unres) == 1 and "2 base materials" in unres.iloc[0]["reason"]


def test_implausible_and_contradictory_rows_are_not_associated():
    df = frame("carrot", ["carrot puree films"], [
        ("(['20.27'], ['cm', '^', '3', '/', 'm', 's', 'Pa'])", "['oxygen', 'barriers']"),              # -> 1.8e14, dropped multiplier
        ("(['0.0639'], ['cm', '^', '3', 'm', '/', 'm', '^', '2', 'day', 'kPa'])", "['OTR']"),          # "OTR" with a coefficient unit
    ])
    mat, unres, _ = associate(df, build_quantities(df))
    assert mat.empty and len(unres) == 2
    assert unres["reason"].str.contains("implausible").sum() == 1 and unres["reason"].str.contains("transmission rate").sum() == 1
    assert PLAUSIBLE_RANGE[GAS_BASE][1] < 1.7e14


def test_placeholder_comparison_never_claims_a_replacement():
    df = frame("pp", ["pure PP"], [PP_O2])
    mat, _, _ = associate(df, build_quantities(df))
    text = compare_with_placeholders(mat, pd.read_csv(MATERIALS_CSV))
    assert "CPP" in text and "NOT replaced" in text and "LDPE" in text and "NO real value" in text

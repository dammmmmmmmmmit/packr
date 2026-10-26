import numpy as np
import pandas as pd
import pytest

from packr.transmat import (audit_transmat, build_doc_summary, build_quantities, clean_unit_tokens,
                            flatten_text, infer_gas, load_transmat, parse_quantity, unit_kind)
from packr.units import GAS_BASE, VAPOUR_BASE


def test_parse_single_range_plusminus_negative():
    assert parse_quantity("(['3400'], ['%'])").value == 3400
    r = parse_quantity("(['0.7', '???', '1.8'], ['%'])")
    assert (r.low, r.high, r.value) == (0.7, 1.8, pytest.approx(1.25))
    pm = parse_quantity("(['23', '??', '1'], ['??', 'C'])")
    assert (pm.value, pm.uncertainty, pm.unit) == (23, 1, "°C")
    assert parse_quantity("(['???', '5'], ['%'])").value == -5
    assert parse_quantity("(['1,200'], ['%'])").value == 1200


@pytest.mark.parametrize("cell", ["not a tuple", None, float("nan"), "(['1'], 'x')", "[1, 2]"])
def test_unparseable_cells_are_flagged_not_crashed(cell):
    assert parse_quantity(cell).status == "unparsed_cell"


def test_unparseable_numbers_are_flagged():
    q = parse_quantity("(['1', 'x', '2', '3'], ['%'])")
    assert q.status == "unparsed_numbers" and q.value is None


def test_unit_token_repair():
    assert clean_unit_tokens(["??", "C"]) == "°C"
    assert clean_unit_tokens(["cm", "^", "3", "??", "m", "/", "(", "m", "^", "2", "day", "kPa", ")"]) \
        == "cm ^ 3 um / ( m ^ 2 day kPa )"
    assert clean_unit_tokens(["m", "^", "???", "2"]) == "m ^ - 2"


def test_unit_kind():
    assert unit_kind("°C") == "temperature_c" and unit_kind("%") == "percent"
    assert unit_kind("um") == "length" and unit_kind("mm") == "length"
    assert unit_kind("cm ^ 3 mm / ( m ^ 2 day atm )") == "permeability_gas"
    assert unit_kind("g mm / ( m ^ 2 day kPa )") == "permeability_vapour"
    assert unit_kind("cm ^ 3 / ( m ^ 2 day atm )") == "other"


def test_infer_gas_checks_co2_before_o2():
    assert infer_gas("permeability", "['carbon', 'dioxide']") == "co2"
    assert infer_gas("co2_permeability") == "co2"
    assert infer_gas("o2_permeability") == "o2"
    assert infer_gas("permeability", "['oxygen']") == "o2"
    assert infer_gas("permeability", "['water', 'vapour']") == "water_vapour"
    assert infer_gas("permeability", None) is None
    assert flatten_text("['a', 'b']") == "a b" and flatten_text(float("nan")) == ""


def test_load_requires_transmat_columns(tmp_path, transmat_file):
    assert len(load_transmat(transmat_file)) == 12
    bad = tmp_path / "bad.tab"
    pd.DataFrame({"a": [1]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="does not look like TRANSMAT"):
        load_transmat(bad)


def test_build_quantities(transmat_df):
    q = build_quantities(transmat_df).set_index("id")
    assert len(q) == 9                                                    # 12 rows minus the 3 SYMBOLIC ones

    # row 0: already in base units, carbon dioxide read from the attached value
    assert q.loc[0, ["gas", "unit_status", "base_unit"]].tolist() == ["co2", "ok", GAS_BASE]
    assert q.loc[0, "value_base"] == pytest.approx(3400)
    # row 1: range 0.7-1.8 cm3 um/(m2 d kPa) -> mid 1.25 * 0.101325
    assert q.loc[1, "gas"] == "o2" and q.loc[1, "value_base"] == pytest.approx(1.25 * 0.101325)
    assert q.loc[1, "low"] == 0.7 and q.loc[1, "high"] == 1.8
    # row 2: water vapour in a mass-based unit
    assert q.loc[2, ["gas", "base_unit"]].tolist() == ["water_vapour", VAPOUR_BASE]
    assert q.loc[2, "value_base"] == pytest.approx(2)
    # row 3: a permeance (no thickness) must NOT be converted
    assert q.loc[3, "unit_status"] == "unit_error" and np.isnan(q.loc[3, "value_base"])
    assert "thickness" in q.loc[3, "unit_note"]
    # rows 4-6: context quantities
    assert q.loc[4, "temperature_c"] == 23 and q.loc[4, "uncertainty"] == 1
    assert q.loc[5, "thickness_um"] == 50
    assert q.loc[6, "unit_kind"] == "percent" and q.loc[6, "unit_status"] == ""
    # row 7: garbage cell is carried through flagged
    assert q.loc[7, "parse_status"] == "unparsed_cell" and q.loc[7, "unit_status"] == "no_value"


def test_gas_unit_family_mismatch_is_rejected(transmat_df):
    bad = transmat_df.copy()
    bad.loc[bad["id"] == 2, "attached_value"] = "['oxygen']"              # oxygen, but a mass-based (vapour) unit
    q = build_quantities(bad).set_index("id")
    assert q.loc[2, "unit_status"] == "family_mismatch" and np.isnan(q.loc[2, "value_base"])


def test_doc_summary(transmat_df):
    q = build_quantities(transmat_df)
    s = build_doc_summary(transmat_df, q).set_index("doc")
    a, b = s.loc["Barrier and surface properties of chitosan-coated films"], s.loc["Barrier properties of chitosan coated polyethylene"]
    assert a["packaging_names"] == "Biopolymer; chitosan coated PE"
    assert a["n_permeability_values"] == 5 and a["n_normalised"] == 3     # 5 rows with a gas, 3 convertible
    assert a["median_p_o2"] == pytest.approx(1.25 * 0.101325)
    assert a["median_p_co2"] == pytest.approx(3400) and a["median_p_water_vapour"] == pytest.approx(2)
    assert a["median_thickness_um"] == 50 and a["temperature_c_min"] == 23
    assert b["median_p_o2"] == pytest.approx(4) and b["packaging_names"] == "PLA-coated paperboard"


def test_audit_reports_failures_and_coverage(transmat_df):
    text = audit_transmat(transmat_df, build_quantities(transmat_df))
    assert "12 rows, 2 documents" in text
    assert "unit_error=" in text and "could NOT be converted" in text
    assert "documents with at least one normalised permeability: 2 of 2" in text


def test_load_tab_separated_by_extension(transmat_file):
    df = load_transmat(transmat_file)                                    # .tab fixture
    assert len(df) == 12 and {"doc", "target", "original_value", "type", "doi"} <= set(df.columns)


def test_load_comma_separated_csv_with_original_capitalised_headers(tmp_path, transmat_df):
    # the CIRAD export as supplied: comma-separated, headers "id, Doc, Target, Original_Value, Attached_Value, Type, Annotator, DOI"
    path = tmp_path / "goldenTRANSMAT-all.csv"
    transmat_df.rename(columns=lambda c: {"doi": "DOI"}.get(c, c[:1].upper() + c[1:]) if c != "id" else c).to_csv(path, index=False)
    df = load_transmat(path)
    assert len(df) == 12 and {"doc", "target", "original_value", "attached_value", "type", "annotator", "doi"} <= set(df.columns)
    assert df.loc[df["id"] == 1, "original_value"].iloc[0].startswith("(['0.7', '???', '1.8']")    # quoted commas survive


def test_load_rejects_unknown_extension(tmp_path, transmat_df):
    path = tmp_path / "goldenTRANSMAT-all.xlsx"
    path.write_text("x")
    with pytest.raises(ValueError, match="unknown TRANSMAT file type"):
        load_transmat(path)


# --- rows copied verbatim from data/raw/goldenTRANSMAT-all.csv (real Unicode: ×, −, ±, –, °) ---------------------------

def real(rows):
    return pd.DataFrame([dict(id=i, doc=d, target=t, original_value=o, attached_value=a, type="QUANTITY", annotator=an, doi="x")
                         for i, an, d, t, o, a in rows])


ID433 = ("(['6.25', '±', '0.20', '×', '10', '^', '−', '11'], "
         "['g', 'm', '^', '−', '1', 's', '^', '−', '1', 'Pa', '^', '−', '1'])")


def test_real_scientific_notation_with_trailing_multiplier_on_both_numbers():
    # id 433: 6.25 ± 0.20 × 10^-11 g m^-1 s^-1 Pa^-1, water vapour (galactomannan coatings)
    q = parse_quantity(ID433)
    assert q.status == "ok" and q.value == pytest.approx(6.25e-11) and q.uncertainty == pytest.approx(0.20e-11)
    assert "both numbers" in q.note
    out = build_quantities(real([(433, 1, "galactomannan", "permeability", ID433, "['water', 'vapour', 'permeability']")]))
    assert out.loc[0, "unit_status"] == "ok" and out.loc[0, "value_base"] == pytest.approx(6.25e-11 * 8.64e10)   # 5.4 g mm/(m2 d kPa)


def test_real_scientific_range():
    # id 82, annotator 2: 0.232 × 10^-10 – 1.160 × 10^-10 g/(m s Pa), WVP
    q = parse_quantity("(['0.232', '×', '10', '^', '−', '10', '–', '1.160', '×', '10', '^', '−', '10'], ['g', '/', 'm', 's', 'Pa'])")
    assert (q.low, q.high) == (pytest.approx(0.232e-10), pytest.approx(1.160e-10)) and q.value == pytest.approx(0.696e-10)


def test_real_positive_exponent_and_bare_power_of_ten():
    assert parse_quantity("(['1.33', '×', '10', '^', '5'], ['Pa'])").value == pytest.approx(1.33e5)          # id 1515
    bare = "(['10', '^', '−', '12'], ['cm', '^', '3', 'cm', '/', 'cm', '^', '2', 's', 'Pa'])"                 # id 1010
    assert parse_quantity(bare).value == pytest.approx(1e-12)


def test_real_table_header_multiplier():
    # id 15: column "P (10^-10 cm3(STP) cm / cm2 s cmHg)": 1.2 means 1.2e-10; CO2 through PLA
    cell = "(['1.2', '(', '10', '^', '−', '10'], ['cm', '^', '3', '(', 'STP', ')', 'cm', '/', 'cm', '^', '2', 's', 'cmHg'])"
    q = parse_quantity(cell)
    assert q.value == pytest.approx(1.2e-10) and "unit header" in q.note
    out = build_quantities(real([(15, 1, "PLA", "permeability", cell, "['carbon', 'dioxide', 'permeation']")]))
    assert out.loc[0, "gas"] == "co2" and out.loc[0, "value_base"] == pytest.approx(1.2 * 65.664, rel=1e-4)   # 1.2 barrer


def test_real_lost_decimal_point_is_not_guessed():
    # id 1068: ['1', '0', '×', '10', '^', '−', '17'] could be 1.0e-17 or 10e-17
    q = parse_quantity("(['1', '0', '×', '10', '^', '−', '17'], ['mol', 'm', '^', '−', '1', 's', '^', '−', '1', 'Pa', '^', '−', '1'])")
    assert q.status == "unparsed_numbers" and q.value is None and "decimal point" in q.note


def test_real_rh_pair():
    q = parse_quantity("(['30', '/', '100'], ['%'])")                                     # id 1532: RH on each side of the film
    assert q.status == "pair" and (q.low, q.high) == (30, 100) and q.value is None


def test_real_degree_sign_as_separate_token_gives_a_temperature():
    # id 1629: (['38'], ['°', 'C']): before the fix every one of the 56 temperatures was missed
    assert clean_unit_tokens(["°", "C"]) == "°C"
    out = build_quantities(real([(1629, 1, "doc", "temperature", "(['38'], ['°', 'C'])", None)]))
    assert out.loc[0, "temperature_c"] == 38


def test_real_thickness_in_nm_and_micrometers():
    out = build_quantities(real([(1666, 1, "doc", "thickness", "(['70'], ['nm'])", "['thickness']"),
                                 (1658, 1, "doc", "thickness", "(['100'], ['micrometers'])", "['thickness']")]))
    assert out["thickness_um"].tolist() == [pytest.approx(0.07), pytest.approx(100)]


def test_real_barrers_plural():
    out = build_quantities(real([(1063, 1, "PP clay", "permeability", "(['2.6'], ['Barrers'])", "['O2']")]))
    assert out.loc[0, "unit_status"] == "ok" and out.loc[0, "value_base"] == pytest.approx(2.6 * 65.664, rel=1e-4)


def test_real_o2_in_a_mass_unit_stays_a_mismatch_with_an_ideal_gas_candidate():
    # id 1025: PO2 6.18 × 10^-17 g/(m s Pa) (chitosan-coated PE). Rejected, but the ideal-gas value is offered for review.
    cell = "(['6.18', '×', '10', '^', '−', '17'], ['g', '/', 'm', 's', 'Pa'])"
    r = build_quantities(real([(1025, 1, "chitosan PE", "permeability", cell, "['PO2']")])).loc[0]
    assert r["unit_status"] == "family_mismatch" and np.isnan(r["value_base"])
    assert r["value_base_ideal_gas"] == pytest.approx(6.18e-17 * 8.64e10 * 22414 / 31.998 * 101.325)
    assert r["base_unit_ideal_gas"] == GAS_BASE


def test_real_water_vapour_in_mol_stays_a_mismatch_with_an_ideal_gas_candidate():
    # id 428: 1.27 ± 0.06 × 10^-11 mol m^-1 s^-1 Pa^-1 water vapour (wheat gluten coated paper)
    cell = "(['1.27', '±', '0.06', '×', '10', '^', '−', '11'], ['mol', 'm', '^', '−', '1', 's', '^', '−', '1', 'Pa', '^', '−', '1'])"
    r = build_quantities(real([(428, 1, "gluten paper", "permeability", cell, "['Water', 'vapour', 'permeability']")])).loc[0]
    assert r["unit_status"] == "family_mismatch"
    assert r["value_base_ideal_gas"] == pytest.approx(1.27e-11 * 18.015 * 1e3 * 86400 * 1e3)                 # mol -> g, Pa -> kPa
    assert r["base_unit_ideal_gas"] == VAPOUR_BASE


def test_real_temperature_annotated_as_permeability_is_flagged():
    out = build_quantities(real([(1, 1, "doc", "permeability", "(['25'], ['°', 'C'])", "['oxygen']")]))
    assert out.loc[0, "unit_status"] == "not_a_permeability"


def test_real_annotation_with_unit_in_the_number_slot_is_no_value():
    # id 4 (chitosan-coated greaseproof paper): the unit tokens sit where the number should be
    cell = "(['cm', '^', '3', 'mm', '/', '(', 'm', '^', '2', 'atm', 'day', ')'], ['carbon', 'dioxide'])"
    assert build_quantities(real([(4, 1, "doc", "permeability", cell, "['permeability']")])).loc[0, "unit_status"] == "no_value"


def test_row_key_is_unique_when_ids_repeat_across_annotators():
    out = build_quantities(real([(82, 2, "a", "temperature", "(['38'], ['°', 'C'])", None),
                                 (82, 3, "a", "temperature", "(['23'], ['°', 'C'])", None)]))
    assert out["row_key"].tolist() == ["2-82", "3-82"]

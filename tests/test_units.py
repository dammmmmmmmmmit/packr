import pytest

from packr.units import (GAS_BASE, VAPOUR_BASE, WVTR_DP_KPA, UnitError, length_to_um, otr_from_permeability,
                         q10_adjust, sat_vapour_pressure_kpa, series_transmission, to_base,
                         wvtr_from_permeability)


def test_base_unit_is_identity():
    assert to_base(3400, "cm^3 mm/(m^2 day atm)") == (3400.0, GAS_BASE)


def test_token_joined_string_as_found_in_transmat():
    # TRANSMAT stores units as token lists; packr joins them with spaces
    assert to_base(10, "cm ^ 3 mm / ( m ^ 2 day atm )")[0] == pytest.approx(10)


def test_micrometre_and_kpa():
    # 1 cm3 um/(m2 d kPa) = 1e-3 mm * 101.325 kPa/atm = 0.101325 base units
    assert to_base(1, "cm^3 µm/(m^2 d kPa)")[0] == pytest.approx(0.101325)


def test_barrer():
    # 1 barrer = 1e-10 cm3(STP) cm /(cm2 s cmHg) = 65.66 cm3 mm/(m2 day atm)
    value, unit = to_base(1e-10, "cm^3 cm/(cm^2 s cmHg)")
    assert unit == GAS_BASE
    assert value == pytest.approx(65.664, rel=1e-3)


def test_mil_per_100_square_inches():
    # 1 cc mil/(100 in2 day atm) = 0.0254 mm / 0.064516 m2 = 0.3937
    assert to_base(1, "cc mil/(100 in^2 day atm)")[0] == pytest.approx(0.3937, rel=1e-3)


def test_negative_exponent_notation():
    assert to_base(5, "cm3 mm m-2 day-1 atm-1")[0] == pytest.approx(5)
    assert to_base(5, "cm^3 mm m^-2 day^-1 atm^-1")[0] == pytest.approx(5)


def test_vapour_family():
    assert to_base(1, "g mm/(m^2 day kPa)") == (1.0, VAPOUR_BASE)
    assert to_base(1, "g mm/(m^2 day atm)")[0] == pytest.approx(1 / 101.325)


def test_barrer_by_name_and_qualifiers_and_compact_forms():
    assert to_base(1, "barrer") == pytest.approx((65.664, GAS_BASE), rel=1e-3)
    assert to_base(1, "Barrer")[0] == pytest.approx(65.664, rel=1e-3)
    assert to_base(1, "cm3(STP) mm m-2 day-1 atm-1")[0] == pytest.approx(1)
    assert to_base(1, "cc·mil/100in2·day·atm")[0] == pytest.approx(0.3937, rel=1e-3)
    assert to_base(1, "cm3 mm/m2 day kPa")[0] == pytest.approx(101.325)
    assert to_base(1, "g·μm/(m²·d·kPa)")[0] == pytest.approx(0.001)
    assert to_base(1, "cm3 mm / (m2 24 h atm)")[0] == pytest.approx(1)


def test_a_power_of_ten_prefix_is_refused_not_silently_dropped():
    # a dropped 10^-10 would be off by ten orders of magnitude
    with pytest.raises(UnitError, match="unparsed text"):
        to_base(1, "10^-10 cm3 cm/(cm2 s cmHg)")


@pytest.mark.parametrize("bad", [
    "cm^3/(m^2 day atm)",            # permeance: no thickness
    "mmol/(m^2 s)",                  # unknown terms
    "%",
    "",
    "m^2 day atm/(cm^3 mm)",         # terms on the wrong side
    "cm^3 g mm/(m^2 day atm)",       # volume and mass together
])
def test_unrecognised_units_raise_instead_of_guessing(bad):
    with pytest.raises(UnitError):
        to_base(1, bad)


def test_no_thickness_message_is_explicit():
    with pytest.raises(UnitError, match="thickness"):
        to_base(1, "cm^3/(m^2 day atm)")


def test_length_to_um():
    assert length_to_um(25, "um") == pytest.approx(25)
    assert length_to_um(0.05, "mm") == pytest.approx(50)
    assert length_to_um(2, "mil") == pytest.approx(50.8)
    with pytest.raises(UnitError):
        length_to_um(1, "kg")


def test_otr_and_wvtr_from_permeability():
    assert otr_from_permeability(0.325, 50) == pytest.approx(6.5)          # P / thickness(mm)
    assert wvtr_from_permeability(1.0, 100, dp_kpa=5.0) == pytest.approx(50.0)


def test_tropical_test_condition():
    assert sat_vapour_pressure_kpa(38) == pytest.approx(6.63, abs=0.05)
    assert WVTR_DP_KPA == pytest.approx(5.96, abs=0.05)


def test_series_and_q10():
    assert series_transmission([2.0, 2.0]) == pytest.approx(1.0)
    assert q10_adjust(10, 5, 15, q10=2.5) == pytest.approx(25.0)
    assert q10_adjust(10, 15, 5, q10=2.5) == pytest.approx(4.0)


# --- units found in the real TRANSMAT file (goldenTRANSMAT-all.csv). Expected values computed by hand:
#     gas base    cm3 mm/(m2 day atm): m^-1 = 1e3 mm/m2, per s = 86400 per day, per Pa = 101325 per atm
#     vapour base g mm/(m2 day kPa):   per Pa = 1000 per kPa

SEC_PER_DAY, PA_PER_ATM = 86400.0, 101325.0


@pytest.mark.parametrize("unit, family, expected", [
    ("g / m s Pa", VAPOUR_BASE, 1e3 * SEC_PER_DAY * 1e3),                                  # g/(m s Pa) = 8.64e10
    ("g m / m ^ 2 s Pa", VAPOUR_BASE, 1e3 * SEC_PER_DAY * 1e3),
    ("g m ^ − 1 s ^ − 1 Pa ^ − 1", VAPOUR_BASE, 1e3 * SEC_PER_DAY * 1e3),                  # Unicode minus
    ("g m ( Pa s m ^ 2 ) ^ − 1", VAPOUR_BASE, 1e3 * SEC_PER_DAY * 1e3),                    # inverted group
    ("ng m / m ^ 2 s Pa", VAPOUR_BASE, 1e-9 * 1e3 * SEC_PER_DAY * 1e3),                    # 86.4
    ("g m / m ^ 2 day kPa", VAPOUR_BASE, 1e3),
    ("g / m day atm", VAPOUR_BASE, 1e3 / 101.325),
    ("g mm / kPa min m ^ 2", VAPOUR_BASE, 1440.0),
    ("cm ^ 3 cm / cm ^ 2 s Pa", GAS_BASE, 1e5 * SEC_PER_DAY * PA_PER_ATM),                 # cm/cm2 = 100 m^-1 = 1e5 mm/m2
    ("cc m / m ^ 2 day atm", GAS_BASE, 1e3),
    ("m ^ 3 m / m ^ 2 s Pa", GAS_BASE, 1e6 * 1e3 * SEC_PER_DAY * PA_PER_ATM),              # m3 = 1e6 cm3
    ("cm ^ 3 / m s Pa", GAS_BASE, 1e3 * SEC_PER_DAY * PA_PER_ATM),
    ("cm ^ 3 m / m ^ 2 day kPa", GAS_BASE, 1e3 * 101.325),
    ("cm ^ 3 μm / m ^ 2 day mmHg", GAS_BASE, 1e-3 * PA_PER_ATM / 133.322387415),         # 0.76
    ("cm ^ 3 ( STP ) cm / cm ^ 2 s cmHg )", GAS_BASE, 1e5 * SEC_PER_DAY * PA_PER_ATM / 1333.22387415),   # stray ")" as in the file
    ("cm ^ 3 ( STP ) cm / cm ^ 2 s cm Hg )", GAS_BASE, 1e5 * SEC_PER_DAY * PA_PER_ATM / 1333.22387415),  # "cm Hg" split in two
    ("mol m ^ − 1 s ^ − 1 Pa ^ − 1", GAS_BASE, 22414.0 * 1e3 * SEC_PER_DAY * PA_PER_ATM),
    ("mol m ^ − 1 Pa ^ − 1 s ^ − 1", GAS_BASE, 22414.0 * 1e3 * SEC_PER_DAY * PA_PER_ATM),  # reordered twin
    ("Barrers", GAS_BASE, 1e-10 * 1e5 * SEC_PER_DAY * PA_PER_ATM / 1333.22387415),
])
def test_transmat_units_convert_to_the_hand_computed_value(unit, family, expected):
    value, base = to_base(1, unit)
    assert base == family and value == pytest.approx(expected, rel=1e-9)


def test_pressure_conversions_are_exact():
    from packr.units import parse_permeability_unit as p
    assert p("cm3 mm/(m2 day Pa)").factor == pytest.approx(101325.0, rel=1e-12)
    assert p("cm3 mm/(m2 day bar)").factor == pytest.approx(101325.0 / 1e5, rel=1e-12)
    assert p("cm3 mm/(m2 day kPa)").factor == pytest.approx(101.325, rel=1e-12)
    assert p("cm3 mm/(m2 day cmHg)").factor == pytest.approx(101325.0 / 1333.22387415, rel=1e-12)
    assert p("cm3 mm/(m2 day torr)").factor == pytest.approx(760.0, rel=1e-12)           # torr = 101325/760 exactly
    assert p("cm3 mm/(m2 day mmHg)").factor != p("cm3 mm/(m2 day torr)").factor          # mmHg is not exactly a torr


def test_barrer_is_exact_with_cmhg():
    assert to_base(1, "barrer")[0] == pytest.approx(1e-10 * 1e5 * 86400 * 101325 / 1333.22387415, rel=1e-12)


def test_mol_units_state_the_molar_volume():
    from packr.units import parse_permeability_unit
    assert "22414" in parse_permeability_unit("mol m ^ − 1 s ^ − 1 Pa ^ − 1").note
    assert parse_permeability_unit("cm3 mm/(m2 day atm)").note == ""


@pytest.mark.parametrize("unit, reason", [
    ("g / m ^ 2 s", "transmission rate"),                     # a WVTR, no thickness
    ("g m ^ − 2 day ^ − 1", "transmission rate"),
    ("kg m ^ − 2 s ^ − 1", "transmission rate"),
    ("cc / m ^ 2 × day", "transmission rate"),
    ("cm ^ 3 / ( m ^ 2 atm day )", "no thickness"),           # permeance
    ("cm ^ 3 / m ^ 2 / 10 ^ 5 Pa / day", "permeance"),        # power of ten AND no thickness
    ("g m ^ − 1 s ^ − 1 Pa", r"pressure\^1"),                 # Pa in the numerator: probably a typo in the source, not guessed
])
def test_transmat_units_that_must_stay_rejected(unit, reason):
    with pytest.raises(UnitError, match=reason):
        to_base(1, unit)


def test_new_plain_lengths():
    assert length_to_um(70, "nm") == pytest.approx(0.07)
    assert length_to_um(100, "micrometers") == pytest.approx(100)
    assert length_to_um(25, "μ m") == pytest.approx(25)


def test_ideal_gas_basis_conversions():
    from packr.units import gas_base_from_vapour_base, vapour_base_from_gas_base
    v, note = gas_base_from_vapour_base(1.0, "o2")                                       # 1 g mm/(m2 day kPa) of O2
    assert v == pytest.approx(22414.0 / 31.998 * 101.325) and "ideal gas" in note
    w, _ = vapour_base_from_gas_base(22414.0 * 101.325)                                  # one mol-equivalent per kPa -> 18.015 g
    assert w == pytest.approx(18.015)

"""Unit handling for barrier properties.

Base units used everywhere in Packr
    gas permeability coefficient P    cm3(STP) mm / (m2 day atm)
    vapour permeability coefficient   g mm / (m2 day kPa)
    OTR                               cm3 / (m2 day) at a 1 atm oxygen difference
    WVTR                              g / (m2 day) at 38 C and 90 % RH (tropical test condition)

Papers report permeability in many unit systems (barrer, cm3 um/(m2 d kPa), cc mil/(100 in2 d atm),
mol m/(m2 s Pa) ...). `to_base` converts what it can recognise and raises UnitError for the rest,
so an unknown unit is never silently guessed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


class UnitError(ValueError):
    """A unit string could not be converted safely."""


GAS_BASE = "cm3 mm / (m2 day atm)"
VAPOUR_BASE = "g mm / (m2 day kPa)"

# Pressure in Pa. Exact by definition: atm 101325, bar 1e5, mmHg 133.322387415 (cmHg = 10 mmHg), torr = 101325/760.
_PA = {"atm": 101325.0, "kpa": 1000.0, "pa": 1.0, "bar": 1.0e5, "mbar": 100.0,
       "cmhg": 1333.22387415, "mmhg": 133.322387415, "torr": 101325.0 / 760.0}
_LENGTH_M = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3, "um": 1.0e-6, "nm": 1.0e-9, "mil": 2.54e-5, "in": 0.0254}
_LENGTH_MM = {"mm": 1.0, "um": 1.0e-3, "cm": 10.0, "m": 1000.0, "mil": 0.0254}     # plain lengths (thickness)
_TIME_DAY = {"day": 1.0, "days": 1.0, "d": 1.0, "h": 1.0 / 24.0, "hr": 1.0 / 24.0, "min": 1.0 / 1440.0, "s": 1.0 / 86400.0}
MOLAR_VOLUME_STP_CM3 = 22414.0          # cm3 of ideal gas per mol at STP (0 C, 1 atm)
_VOLUME_CM3 = {"cc": 1.0, "ml": 1.0, "mol": MOLAR_VOLUME_STP_CM3}
_MASS_G = {"g": 1.0, "mg": 1.0e-3, "ug": 1.0e-6, "ng": 1.0e-9, "kg": 1.0e3}

_ATOM = re.compile(r"(?P<coef>\d+(?:\.\d+)?)?\s*(?P<name>[a-z]+)(?:\^(?P<exp>-?\d+))?")
_GROUP_INVERSE = re.compile(r"\(([^()/]*)\)\s*\^-1\b")
_QUALIFIERS = {"stp", "ntp"}     # "cm3 (STP)" states the reference gas condition; it carries no factor
_BARRER_UNIT = "cm^3 cm/(cm^2 s cmhg)"   # 1 barrer = 1e-10 of this


def _negate(atom: str) -> str:
    m = _ATOM.fullmatch(atom)
    if not m:
        return atom
    exp = -int(m.group("exp") or 1)
    return f"{m.group('coef') or ''}{m.group('name')}^{exp}"


def _normalise(text: str) -> str:
    t = text.strip().lower()
    for old, new in (("µ", "u"), ("μ", "u"), ("·", " "), ("⋅", " "), ("×", " "), ("*", " "),
                     ("−", "-"), ("–", "-"), ("²", "^2"), ("³", "^3"),
                     ("⁻¹", "^-1"), ("⁻²", "^-2"), ("⁻³", "^-3")):
        t = t.replace(old, new)
    t = re.sub(r"\s*\^\s*", "^", t)                  # "cm ^ 3" -> "cm^3"
    t = re.sub(r"\^-\s+(\d)", r"^-\1", t)            # "m^- 2"  -> "m^-2"
    t = re.sub(r"\b(c?m)\s+hg\b", r"\1hg", t)        # "cm Hg"  -> "cmhg"
    t = re.sub(r"(?<![a-z])u\s+(m|mol)\b", r"u\1", t)   # "μ m" (split token) -> "um"
    t = re.sub(r"(?<![a-z])(cm|mm|m|in)([23])\b", r"\1^\2", t)  # "cm3" -> "cm^3", "m2" -> "m^2", "100in2" -> "100in^2"
    t = re.sub(r"([a-z])-(\d)", r"\1^-\2", t)        # "atm-1" -> "atm^-1"
    # "(Pa s m^2)^-1" -> "pa^-1 s^-1 m^-2": invert every term of a parenthesised group raised to -1
    t = _GROUP_INVERSE.sub(lambda m: " " + " ".join(_negate(a) for a in m.group(1).split()) + " ", t)
    return t.replace("(", " ").replace(")", " ")


@dataclass(frozen=True)
class Permeability:
    family: str        # "gas" or "vapour"
    factor: float      # multiply a value in the source unit by this to get the base unit
    base_unit: str
    note: str = ""     # assumptions used in the conversion (e.g. the molar volume for mol-based units)


def parse_permeability_unit(text: str) -> Permeability:
    """Parse a permeability *coefficient* unit by dimensional analysis.

    Every term is reduced to one of: gas amount (cm3 STP: a length cubed in the numerator, cc, ml, or mol at 22414 cm3/mol),
    vapour amount (g), length (m), time (day) or pressure (Pa). A coefficient must come out as amount^1 length^-1 time^-1
    pressure^-1 (thickness / area). Net length^-2 is a permeance or transmission rate: rejected, never guessed.
    """
    if not isinstance(text, str) or not text.strip():
        raise UnitError("empty unit")
    normalised = _normalise(text)
    if normalised.strip() in ("barrer", "barrers"):
        return Permeability("gas", 1e-10 * parse_permeability_unit(_BARRER_UNIT).factor, GAS_BASE)
    amounts: list[tuple[str, int, float]] = []            # (family, exponent, factor)
    length_exp = time_exp = pressure_exp = 0
    scale = 1.0                                           # product of length (m), time (day) and pressure (Pa) factors
    notes: list[str] = []
    for index, segment in enumerate(normalised.split("/")):
        side = 1 if index == 0 else -1                    # everything after a "/" is in the denominator
        leftover = _ATOM.sub(" ", segment)
        if re.search(r"[0-9a-z]", leftover):              # e.g. a "10^-10" prefix: refuse rather than drop it silently
            extra = ""
            if re.search(r"\b10\^", normalised) and not re.search(r"(?<![a-z])(m|cm|mm|um|mil)(\^1)?\b(?!\^)", normalised.split("/")[0]):
                extra = "; it also has no thickness term in the numerator, so it is a permeance, not a coefficient"
            raise UnitError(f"unparsed text '{leftover.strip()}' in '{text}' "
                            f"(a power-of-ten multiplier inside a unit is not supported{extra})")
        for m in _ATOM.finditer(segment):
            name = m.group("name")
            if name in _QUALIFIERS:
                continue
            coef = float(m.group("coef")) if m.group("coef") else 1.0
            exp = side * (int(m.group("exp")) if m.group("exp") else 1)
            if name in _LENGTH_M and exp == 3:
                amounts.append(("gas", 1, coef * (_LENGTH_M[name] * 100.0) ** 3))     # a volume: m^3 = 1e6 cm3
            elif name in _LENGTH_M and exp != -3:
                length_exp += exp
                scale *= (coef * _LENGTH_M[name] ** abs(exp)) ** (1 if exp > 0 else -1)
            elif name in _VOLUME_CM3 and abs(exp) == 1:
                amounts.append(("gas", exp, coef * _VOLUME_CM3[name]))
                if name == "mol":
                    notes.append(f"mol converted at the STP molar volume {MOLAR_VOLUME_STP_CM3:g} cm3/mol")
            elif name in _MASS_G and abs(exp) == 1:
                amounts.append(("vapour", exp, coef * _MASS_G[name]))
            elif name in _TIME_DAY and abs(exp) == 1:
                time_exp += exp
                scale *= (coef * _TIME_DAY[name]) ** exp
            elif name in _PA and abs(exp) == 1:
                pressure_exp += exp
                scale *= (coef * _PA[name]) ** exp
            else:
                raise UnitError(f"unrecognised unit term '{name}^{exp}' in '{text}'")

    if len(amounts) != 1:
        kinds = ", ".join(sorted({a[0] for a in amounts})) or "none"
        raise UnitError(f"need exactly one gas-volume or mass term in '{text}', found {len(amounts)} ({kinds})")
    family, amount_exp, amount = amounts[0]
    if amount_exp != 1:
        raise UnitError(f"terms of '{text}' are on the wrong side of the fraction (amount in the denominator)")
    if pressure_exp == 0 and length_exp == -2 and time_exp == -1:
        raise UnitError(f"no thickness or pressure term in '{text}': a transmission rate per area (e.g. WVTR), "
                        "not a permeability coefficient")
    if length_exp == -2:
        raise UnitError(f"no thickness term in '{text}': a transmission rate or permeance (per area), "
                        "not a permeability coefficient")
    if time_exp != -1 or pressure_exp != -1 or length_exp != -1:
        raise UnitError(f"'{text}' is not a permeability coefficient: length^{length_exp} time^{time_exp} pressure^{pressure_exp} "
                        "(need length^-1 time^-1 pressure^-1)")
    # scale is in m^-1 day^-1 Pa^-1; the base units are mm/m2 (= 1e-3 m^-1) per day per atm (gas) or per kPa (vapour)
    per_pressure = 101325.0 if family == "gas" else 1000.0
    return Permeability(family, amount * scale * per_pressure / 1e-3, GAS_BASE if family == "gas" else VAPOUR_BASE, "; ".join(notes))


def to_base(value: float, unit: str) -> tuple[float, str]:
    """Convert a permeability coefficient to Packr's base unit. Returns (value, base_unit)."""
    perm = parse_permeability_unit(unit)
    return value * perm.factor, perm.base_unit


_LENGTH_WORDS = {"micrometer": "um", "micrometers": "um", "micrometre": "um", "micrometres": "um", "micron": "um",
                 "microns": "um", "nm": "nm", "nanometer": "nm", "nanometers": "nm", "millimeter": "mm", "millimeters": "mm"}


def length_to_um(value: float, unit: str) -> float:
    """Convert a plain length (nm, um / micrometers, mm, mil, cm, m) to micrometres."""
    key = _normalise(unit).strip()
    key = _LENGTH_WORDS.get(key, key)
    if key == "nm":
        return value * 1.0e-3
    if key not in _LENGTH_MM:
        raise UnitError(f"'{unit}' is not a recognised length unit")
    return value * _LENGTH_MM[key] * 1000.0


# Converting between a mass basis (g) and a gas-volume basis (cm3 STP) for a named gas assumes an ideal gas:
# 1 g = MOLAR_VOLUME_STP_CM3 / M cm3(STP). Molar masses in g/mol.
MOLAR_MASS_G = {"o2": 31.998, "co2": 44.009, "n2": 28.014, "water_vapour": 18.015}


def gas_base_from_vapour_base(value: float, gas: str) -> tuple[float, str]:
    """A permeability of O2 / CO2 / N2 reported on a mass basis (g mm/(m2 day kPa)) -> cm3 mm/(m2 day atm)."""
    m = MOLAR_MASS_G[gas]
    note = f"mass basis converted to cm3(STP) with M({gas}) = {m:g} g/mol and {MOLAR_VOLUME_STP_CM3:g} cm3/mol (ideal gas)"
    return value * MOLAR_VOLUME_STP_CM3 / m * 101.325, note


def vapour_base_from_gas_base(value: float) -> tuple[float, str]:
    """Water vapour reported on a mol or cm3(STP) basis (cm3 mm/(m2 day atm)) -> g mm/(m2 day kPa)."""
    m = MOLAR_MASS_G["water_vapour"]
    note = f"mol / cm3(STP) basis converted to g with M(H2O) = {m:g} g/mol and {MOLAR_VOLUME_STP_CM3:g} cm3/mol (ideal gas)"
    return value / MOLAR_VOLUME_STP_CM3 * m / 101.325, note


def sat_vapour_pressure_kpa(temp_c: float) -> float:
    """Saturation vapour pressure of water (Tetens equation), kPa."""
    return 0.61078 * math.exp(17.27 * temp_c / (temp_c + 237.3))


WVTR_DP_KPA = 0.90 * sat_vapour_pressure_kpa(38.0)    # driving force at the 38 C / 90 % RH test condition


def otr_from_permeability(p_gas: float, thickness_um: float) -> float:
    """OTR (cm3/m2/day at 1 atm) of a homogeneous film from its permeability coefficient."""
    return p_gas / (thickness_um / 1000.0)


def wvtr_from_permeability(p_vapour: float, thickness_um: float, dp_kpa: float = WVTR_DP_KPA) -> float:
    """WVTR (g/m2/day) of a homogeneous film from its vapour permeability coefficient."""
    return p_vapour * dp_kpa / (thickness_um / 1000.0)


def series_transmission(rates: list[float]) -> float:
    """Transmission rate of layers in series: 1 / sum(1 / rate_i)."""
    return 1.0 / sum(1.0 / r for r in rates)


def q10_adjust(rate_ref: float, t_ref_c: float, t_c: float, q10: float = 2.5) -> float:
    """Scale a respiration rate measured at t_ref_c to t_c (Q10 ~ 2-3 for most produce)."""
    return rate_ref * q10 ** ((t_c - t_ref_c) / 10.0)

"""Reshape the TRANSMAT annotation table into something usable.

TRANSMAT (CIRAD Dataverse, doi:10.18167/DVN1/U7HK8J, file goldenTRANSMAT-all.tab) is entity-extraction
data from ~50 papers. Each row is ONE annotated entity (a quantity or a name), not one material, with columns

    id, doc, target, original_value, attached_value, type, annotator, doi

* type == "QUANTITY": original_value is the repr of ([number tokens], [unit tokens]),
  e.g. (['0.7', '???', '1.8'], ['cm', '^', '3', 'mm', ...])
* type == "SYMBOLIC": original_value is the repr of a token list, e.g. ['chitosan', 'coated', 'PE']

The export lost non-ASCII bytes (one '?' per byte), so en dashes, +/-, degree and micro signs arrive as
'???' / '??'. `clean_unit_tokens` and `_parse_numbers` repair them from context. That repair is a heuristic
based on the rows seen so far: check the audit output before trusting it.

IMPORTANT LIMIT: the table does not say which quantity belongs to which material. Everything here is
reliable per row and per document; linking a permeability value to one named material inside a document
is the open problem (see ROADMAP.md).
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import (GAS_BASE, MOLAR_MASS_G, VAPOUR_BASE, UnitError, gas_base_from_vapour_base, length_to_um,
                    parse_permeability_unit, vapour_base_from_gas_base)

REQUIRED_COLUMNS = {"doc", "target", "original_value", "type"}
_RANGE = {"???", "–", "—", "-", "to"}
_PLUS_MINUS = {"??", "±", "+/-"}
_MINUS = {"???", "-", "−"}


_SEPARATORS = {".tab": "\t", ".tsv": "\t", ".csv": ","}


def load_transmat(path: str | Path) -> pd.DataFrame:
    """Read goldenTRANSMAT-all (.tab / .tsv tab-separated, .csv comma-separated) and normalise column names."""
    suffix = Path(path).suffix.lower()
    if suffix not in _SEPARATORS:
        raise ValueError(f"{path}: unknown TRANSMAT file type '{suffix}'; expected one of {sorted(_SEPARATORS)}")
    df = None
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=_SEPARATORS[suffix], encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    assert df is not None
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} does not look like TRANSMAT: missing {sorted(missing)}; found {list(df.columns)}")
    return df


def _literal(cell):
    if not isinstance(cell, str):
        return None
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return None


def flatten_text(cell) -> str:
    """"['carbon', 'dioxide']" -> 'carbon dioxide'. Plain strings pass through, NaN -> ''."""
    if not isinstance(cell, str):
        return ""
    obj = _literal(cell)
    return " ".join(str(t) for t in obj) if isinstance(obj, list) else cell


@dataclass
class Quantity:
    value: float | None = None
    low: float | None = None
    high: float | None = None
    uncertainty: float | None = None
    unit: str = ""
    status: str = "ok"          # ok | pair | unparsed_cell | unparsed_numbers
    note: str = ""              # how an unusual number pattern was read, or why it was not


def _num(token) -> float | None:
    try:
        v = float(str(token).replace(",", ""))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


_TIMES = {"×", "x", "*"}


def _scalar(t: list[str], i: int) -> tuple[float, int | None, int] | None:
    """Read 'a', 'a × 10 ^ [−] n' or '10 ^ [−] n' starting at t[i]. Returns (mantissa, power or None, next index)."""
    def power(j: int) -> tuple[int, int] | None:            # t[j] == "10", t[j+1] == "^", optional minus, integer
        if t[j:j + 2] == ["10", "^"]:
            k, sign = j + 2, 1
            if k < len(t) and t[k] in _MINUS:
                sign, k = -1, k + 1
            if k < len(t) and re.fullmatch(r"\d+", t[k]):
                return sign * int(t[k]), k + 1
        return None

    if i >= len(t):
        return None
    if (pw := power(i)) is not None:                         # bare "10 ^ − 12"
        return 1.0, pw[0], pw[1]
    a = _num(t[i])
    if a is None:
        return None
    if i + 1 < len(t) and t[i + 1] in _TIMES and (pw := power(i + 2)) is not None:
        return a, pw[0], pw[1]
    return a, None, i + 1


def _parse_numbers(tokens: list) -> Quantity:
    t = [str(x).strip() for x in tokens]
    if len(t) == 2 and t[0] in _MINUS and (v := _num(t[1])) is not None:
        return Quantity(value=-v)
    first = _scalar(t, 0)
    if first is None:
        return Quantity(status="unparsed_numbers", note="does not start with a number")
    a, pa, i = first
    if i == len(t):
        return Quantity(value=a * 10.0 ** (pa or 0))
    # table-header multiplier: "0.04 ( 10 ^ − 10" from a column headed "P (10^-10 cm3(STP) cm / cm2 s cmHg)"
    if pa is None and t[i] == "(" and (hdr := _scalar(t, i + 1)) is not None and hdr[0] == 1.0 and hdr[1] is not None and hdr[2] == len(t):
        return Quantity(value=a * 10.0 ** hdr[1], note=f"multiplier 10^{hdr[1]} taken from the unit header '(10^{hdr[1]} ...'")
    sep = t[i]
    if pa is None and _num(sep) is not None and i + 1 < len(t) and t[i + 1] in _TIMES:
        return Quantity(status="unparsed_numbers",
                        note=f"two numbers '{a:g} {sep}' before the multiplier: a decimal point was probably lost "
                             f"({a:g}.{sep} or {t[0]}{sep}?); not guessed")
    second = _scalar(t, i + 1)
    if second is None or second[2] != len(t):
        return Quantity(status="unparsed_numbers", note=f"unrecognised number pattern {' '.join(t)!r}")
    b, pb, _ = second
    note = ""
    if pa is None and pb is not None:                        # "0.55 ± 0.05 × 10^-5" means (0.55 ± 0.05) × 10^-5
        pa, note = pb, f"trailing multiplier 10^{pb} applied to both numbers"
    a, b = a * 10.0 ** (pa or 0), b * 10.0 ** (pb or 0)
    if sep in _PLUS_MINUS:
        return Quantity(value=a, uncertainty=abs(b), note=note)
    if sep in _RANGE:
        lo, hi = sorted((a, b))
        return Quantity(value=(lo + hi) / 2, low=lo, high=hi, note=note)
    if sep == "/" and pa is None and pb is None:             # relative humidity on the two sides of a film: "0 / 33" %
        return Quantity(low=a, high=b, status="pair", note="two values (e.g. RH on each side of the film)")
    return Quantity(status="unparsed_numbers", note=f"unrecognised separator {sep!r}")


def clean_unit_tokens(tokens: list) -> str:
    """Join unit tokens into one string, repairing '?' placeholders and split degree / micro signs from context."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = str(tokens[i])
        nxt = str(tokens[i + 1]) if i + 1 < len(tokens) else ""
        if tok in ("??", "°") and nxt == "C":          # degree sign, damaged ('??') or as a separate token ('°', 'C')
            out.append("°C")
            i += 2
        elif tok in ("??", "μ", "µ") and nxt.startswith("m"):   # micro sign: ?? m -> um, μ mol -> umol
            out.append("u" + nxt)
            i += 2
        elif tok == "???":                             # minus sign inside an exponent
            out.append("-")
            i += 1
        else:
            out.append(tok)
            i += 1
    return " ".join(out)


def parse_quantity(cell) -> Quantity:
    obj = _literal(cell)
    if not (isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, list) for x in obj)):
        return Quantity(status="unparsed_cell")
    q = _parse_numbers(obj[0])
    q.unit = clean_unit_tokens(obj[1])
    return q


def infer_gas(*cells) -> str | None:
    """co2 / o2 / water_vapour from a target name and its attached value. 'co2' is tested first because it contains 'o2'."""
    text = " ".join(flatten_text(c) if isinstance(c, str) else "" for c in cells).lower()
    if "co2" in text or "carbon dioxide" in text:
        return "co2"
    if "h2o" in text or "water" in text or "vapour" in text or "vapor" in text or re.search(r"\bwv(p|tr)\b", text):
        return "water_vapour"
    # "OP", "OP-values", "OTR" are the papers' abbreviations for oxygen permeability / transmission rate
    if "o2" in text or "oxygen" in text or re.search(r"\b(op|otr)\b", text):
        return "o2"
    return None


def unit_kind(unit: str) -> str:
    u = unit.strip().lower()
    if u in {"°c", "c", "degc"}:
        return "temperature_c"
    if u == "%":
        return "percent"
    try:
        length_to_um(1, u)
        return "length"
    except UnitError:
        pass
    try:
        return "permeability_" + parse_permeability_unit(unit).family
    except UnitError:
        return "other"


def _factor_kind(text: str) -> str:
    t = text.lower()
    if "thick" in t:
        return "thickness"
    if "temp" in t:
        return "temperature"
    if "humid" in t or re.search(r"\brh\b", t):
        return "humidity"
    if "pressure" in t:
        return "pressure"
    return ""


def build_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """One row per QUANTITY entity, parsed, with permeabilities converted to Packr's base units."""
    rows = []
    for r in df[df["type"].astype(str).str.upper() == "QUANTITY"].to_dict("records"):
        q = parse_quantity(r.get("original_value"))
        target = str(r.get("target", ""))
        attached = flatten_text(r.get("attached_value"))
        kind = unit_kind(q.unit) if q.status == "ok" else ""
        is_perm = "perm" in target.lower()
        gas = infer_gas(target, attached) if is_perm else None
        value_base, base_unit, status, note = np.nan, "", "", ""
        # A gas named O2/CO2 in a mass unit (or water vapour in mol) stays "family_mismatch": it may be a mislabelled row. The
        # ideal-gas conversion is offered in value_base_ideal_gas for review only; nothing downstream uses it.
        value_ideal, unit_ideal = np.nan, ""
        if is_perm:
            if q.value is None:
                status, note = "no_value", q.note or q.status
            elif kind in ("temperature_c", "length", "percent"):
                status, note = "not_a_permeability", f"annotated as a permeability but the unit '{q.unit}' is a {kind}"
            else:
                try:
                    perm = parse_permeability_unit(q.unit)
                    value_base, base_unit, note = q.value * perm.factor, perm.base_unit, perm.note
                    family = perm.family
                    gas = gas or ("water_vapour" if family == "vapour" else "unspecified_gas")
                    status = "ok"
                    if (gas == "water_vapour") != (family == "vapour"):
                        if gas in MOLAR_MASS_G and family == "vapour":
                            value_ideal, extra = gas_base_from_vapour_base(value_base, gas)
                            unit_ideal = GAS_BASE
                        elif gas == "water_vapour":
                            value_ideal, extra = vapour_base_from_gas_base(value_base)
                            unit_ideal = VAPOUR_BASE
                        else:
                            extra = "gas not named, no conversion possible"
                        value_base, base_unit, status = np.nan, "", "family_mismatch"
                        note = f"gas '{gas}' but unit is a {family} unit; ideal-gas candidate: {extra}"
                except UnitError as exc:
                    status, note = "unit_error", str(exc)
        factor = _factor_kind(f"{attached} {target}")
        rows.append({
            "row_key": f"{r.get('annotator')}-{r.get('id')}",    # TRANSMAT ids repeat across annotators; (annotator, id) is unique
            "id": r.get("id"), "doc": r.get("doc"), "doi": r.get("doi"), "annotator": r.get("annotator"),
            "target": target, "attached": attached, "factor": factor, "gas": gas,
            "value": q.value, "low": q.low, "high": q.high, "uncertainty": q.uncertainty,
            "unit": q.unit, "unit_kind": kind, "parse_status": q.status, "parse_note": q.note,
            "value_base": value_base, "base_unit": base_unit, "unit_status": status, "unit_note": note,
            "value_base_ideal_gas": value_ideal, "base_unit_ideal_gas": unit_ideal,
            "thickness_um": length_to_um(q.value, q.unit) if (kind == "length" and factor == "thickness"
                                                                and q.value is not None) else np.nan,
            "temperature_c": q.value if kind == "temperature_c" and q.value is not None else np.nan,
        })
    return pd.DataFrame(rows)


def build_doc_summary(df: pd.DataFrame, quantities: pd.DataFrame) -> pd.DataFrame:
    """One row per paper. Document-level only: values are NOT yet tied to individual materials."""
    docs = pd.DataFrame(index=pd.Index(df["doc"].dropna().unique(), name="doc"))
    if "doi" in df.columns:
        docs["doi"] = df.groupby("doc")["doi"].first()

    sym = df[(df["type"].astype(str).str.upper() == "SYMBOLIC")
             & (df["target"].astype(str).str.lower() == "packaging")]
    docs["packaging_names"] = sym.assign(n=sym["original_value"].map(flatten_text)).groupby("doc")["n"].agg(
        lambda s: "; ".join(sorted({x for x in s if x})))

    if len(quantities):
        perm = quantities[quantities["gas"].notna()]
        docs["n_permeability_values"] = perm.groupby("doc").size()
        ok = perm[perm["unit_status"] == "ok"]
        docs["n_normalised"] = ok.groupby("doc").size()
        for gas, grp in ok.groupby("gas"):
            docs[f"median_p_{gas}"] = grp.groupby("doc")["value_base"].median()
        docs["median_thickness_um"] = quantities.groupby("doc")["thickness_um"].median()
        docs["temperature_c_min"] = quantities.groupby("doc")["temperature_c"].min()
        docs["temperature_c_max"] = quantities.groupby("doc")["temperature_c"].max()
    for col in ("n_permeability_values", "n_normalised"):
        docs[col] = docs[col].fillna(0).astype(int) if col in docs else 0
    return docs.reset_index()


def audit_transmat(df: pd.DataFrame, quantities: pd.DataFrame) -> str:
    lines = [f"TRANSMAT: {len(df)} rows, {df['doc'].nunique()} documents",
             "rows by type:   " + ", ".join(f"{k}={v}" for k, v in df["type"].value_counts().items()),
             "rows by target: " + ", ".join(f"{k}={v}" for k, v in df["target"].value_counts().items())]
    if quantities.empty:
        return "\n".join(lines + ["no QUANTITY rows found"])
    lines.append("quantity parsing: " + ", ".join(f"{k}={v}" for k, v in quantities["parse_status"].value_counts().items()))
    perm = quantities[quantities["unit_status"] != ""]
    lines.append(f"permeability rows: {len(perm)}; unit_status: "
                 + ", ".join(f"{k}={v}" for k, v in perm["unit_status"].value_counts().items()))
    ok = perm[perm["unit_status"] == "ok"]
    if len(ok):
        lines.append("normalised values by gas (median, base units): "
                     + ", ".join(f"{g}: n={len(s)} median={s['value_base'].median():.4g}"
                                 for g, s in ok.groupby("gas")))
    lines.append(f"converted: {len(ok)} of {len(perm)} permeability rows ({len(ok) / max(len(perm), 1):.0%})")
    failed = perm[perm["unit_status"] != "ok"]
    if len(failed):
        top = failed["unit"].fillna("").value_counts().head(10)
        lines.append("most common units that could NOT be converted (add support or fix the parse):")
        lines += [f"    {n:>4} x  '{u}'" for u, n in top.items()]
        lines.append("every unconverted permeability row, grouped by status and unit, with the reason:")
        reason = failed["unit_note"].fillna("").astype(str).str.replace(r" in '[^']*'", "", regex=True)
        grp = failed.assign(reason=reason, unit=failed["unit"].fillna("")).groupby(["unit_status", "unit", "reason"]).size()
        for (status, unit, why), n in grp.sort_values(ascending=False).items():
            lines.append(f"    {n:>4} x  [{status}] '{unit}': {why}")
    lines.append(f"documents with at least one normalised permeability: {ok['doc'].nunique()} of {df['doc'].nunique()}")
    lines.append("thickness values found: " + str(int(quantities['thickness_um'].notna().sum()))
                 + "; temperature values found: " + str(int(quantities['temperature_c'].notna().sum())))
    return "\n".join(lines)

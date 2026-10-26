"""Link TRANSMAT permeability values to packaging materials (spec T1, the "hard problem").

TRANSMAT annotates entities separately: it never says which value belongs to which material, and entity ids are grouped by
entity type, not by position in the paper, so id proximity carries no information. The only association made here is:

  single-material paper   every packaging mention in the paper maps (via MATERIAL_ALIASES) to ONE base material. Fillers,
                          plasticisers and generic words ("film", "membrane", "composite") are ignored. All the paper's
                          converted values are then values of that base material, but WHICH variant (neat, +5 % clay,
                          plasticised ...) is not known: `variant_resolved` is always False.

Every other paper goes to the unresolved table with its packaging mentions, for manual association from the paper itself.
Values outside PLAUSIBLE_RANGE are not associated (they are almost always a dropped multiplier or a mis-annotated unit).
Nothing here is guessed from value magnitudes, and no association is invented.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .transmat import flatten_text
from .units import GAS_BASE, VAPOUR_BASE

# Screening bounds, orders of magnitude only (base units). Gas: dry EVOH ~1e-3, silicone rubber ~4e4 cm3 mm/(m2 day atm).
# Vapour: PVdC ~1e-3, very hydrophilic edible films ~1e2 g mm/(m2 day kPa). A value outside is flagged, never corrected.
PLAUSIBLE_RANGE = {GAS_BASE: (1e-5, 1e5), VAPOUR_BASE: (1e-4, 1e3)}

# Words removed before matching base materials: additives, fillers and plasticisers (a composite keeps its matrix as base).
ADDITIVES = [r"polyethylene glycol", r"\bpeg\b", r"glycerol", r"sorbitol", r"fatty acids?", r"\bwater\b", r"nano ?clay", r"\bclay\b",
             r"cloisite( \w+)?", r"\bommt\b", r"\bmmt\b", r"organoclay", r"montmorillonites?", r"layered silicates?", r"\bsilica\b",
             r"1 - decyl[^;]*salts", r"imidazolium( salts)?", r"\b2c18\b", r"ammonium", r"\bc(6|15|20|30)a\b", r"\bna \^ \+",
             r"α-tocopherol", r"\bbht\b", r"cellulose nanocrystals?", r"\b\d*s?-?cnc\b", r"\btio2\b", r"\bzrp\b", r"\bsic\b",
             r"silicon carbide", r"boron nitride", r"\bbn\b"]
# Base materials: (regex on the lower-cased mention after removing additives, base material, family). First match wins per span.
MATERIAL_ALIASES = [
    (r"poly \( lactic acid \)|polylactic acid|polylactide|poly \( lactide \)|\bplas?\b", "PLA", "PLA"),
    (r"poly \( n-butyl methacrylate \)|\bs?pbma\b", "PBMA", "acrylic"),
    (r"polyurethane|\bpu\b", "polyurethane", "PU"),
    (r"\bpbi\b", "polybenzimidazole", "PBI"),
    (r"poly \( ethylene terephthalate \)|\bo?pet\b|\bpetp\b", "PET", "PET"),
    (r"\bevoh\b", "EVOH", "EVOH"),
    (r"linear low-density|\bm?lldpe\b", "LLDPE", "PE"),
    (r"low.density polyethylene|\bldpe\b", "LDPE", "PE"),
    (r"high.density polyethylene|\bhdpe\b", "HDPE", "PE"),
    (r"polyethylene|\bpe\b", "PE (grade not stated)", "PE"),
    (r"polypropylene|\bpp\b", "PP (grade not stated)", "PP"),
    (r"poly \( vinyl chloride \)|polyvinyl chloride|\bpvc\b", "PVC", "PVC"),
    (r"polystyrene|\bps\b", "PS", "PS"),
    (r"nylon|polyamide|\bny\b|\bpa6?\b", "PA (nylon)", "PA"),
    (r"\bsiox\b", "SiOx coating", "inorganic coating"),
    (r"chitosan|\bcs[ae]\b|\bpecse\b", "chitosan", "polysaccharide"),
    (r"methylcellulose|\bmc\b|cellulose-derivative", "methylcellulose", "cellulose derivative"),
    (r"cellophane", "cellophane", "cellulose"),
    (r"starch|amylose", "starch", "polysaccharide"),
    (r"galactomannan|locust bean gum", "galactomannan", "polysaccharide"),
    (r"psyllium.seed gum|psyllium", "psyllium seed gum", "polysaccharide"),
    (r"soluble soybean polysaccharide|\bssps\b", "soluble soybean polysaccharide", "polysaccharide"),
    (r"gelatin", "fish gelatin", "protein"),
    (r"wheat gluten|gluten|\bwg\b", "wheat gluten", "protein"),
    (r"zein|\bcz\b", "corn zein", "protein"),
    (r"soy protein isolates?|soy protein|\bspi\b", "soy protein isolate", "protein"),
    (r"carrot", "carrot puree", "fruit/vegetable puree"),
    (r"paper", "paper", "paper"),
]
# A mention that, after removing additives, matches none of these and still contains a word other than these is "unmapped".
GENERIC_WORDS = {"film", "films", "coating", "coatings", "coated", "membrane", "membranes", "composite", "composites", "nanocomposite",
                 "nanocomposites", "biopolymer", "polymer", "the", "sample", "edible", "blend", "blends", "blended", "bilayer",
                 "material", "high", "thermoplastic", "pure", "neat", "native", "unfilled", "untreated", "uncoated", "based", "protein",
                 "protein-based", "solution", "solutions", "forming", "ffs", "with", "on", "of", "in", "by", "and", "plasticized",
                 "plasticizers", "sonicated", "stirred", "cast", "extruded", "oat", "amorphous", "biaxially", "oriented", "linear",
                 "hybrid", "influenced", "casting", "solvent", "addition", "formulation", "-", "/", "(", ")", "%", "100", "2", "5",
                 "fish", "corn", "layer", "laminated", "metalocene", "catalyzed", "low-density", "treated", "modified", "dialkyl",
                 "thermally-stable", "puree", "peel", "papers", "greaseproof", "type", "s", "a", "–", "+", "^", "×", "pristine",
                 "polyether", "samples", "fractionated", "fractions", "reinforced", "resin", "matrix", "mixed", "crosslinked",
                 "cross-linked", "biodegradable", "compostable", "packaging", "bags", "plastic", "bio", "biobased", "nano",
                 "biocomposite", "biocomposites", "nanobiocomposites", "bionanocomposites", "from", "using", "un"}


def canonical(mention: str) -> tuple[set[str], set[str], str]:
    """(base materials, families, unmapped leftover text) for one packaging mention."""
    t = f" {mention.lower()} "
    for pat in ADDITIVES:
        t = re.sub(pat, " ", t)
    bases, families = set(), set()
    for pat, base, fam in MATERIAL_ALIASES:
        if re.search(pat, t):
            bases.add(base)
            families.add(fam)
            t = re.sub(pat, " ", t)
    leftover = " ".join(w for w in re.split(r"[\s\-–/()]+", t) if w and w not in GENERIC_WORDS and not re.fullmatch(r"[\d.]+", w))
    return bases, families, leftover


def doc_materials(df: pd.DataFrame) -> pd.DataFrame:
    """One row per document: its packaging mentions, the base materials they map to, and anything unmapped."""
    sym = df[(df["type"].astype(str).str.upper() == "SYMBOLIC") & (df["target"].astype(str).str.lower() == "packaging")]
    rows = []
    for doc, grp in sym.groupby("doc"):
        names = sorted({flatten_text(v) for v in grp["original_value"] if flatten_text(v)})
        bases, fams, unmapped = set(), set(), []
        for n in names:
            b, f, left = canonical(n)
            bases |= b
            fams |= f
            if left:
                unmapped.append(f"{n} -> '{left}'")
        rows.append(dict(doc=doc, mentions="; ".join(names), n_mentions=len(names), base_materials="; ".join(sorted(bases)),
                         n_base_materials=len(bases), families="; ".join(sorted(fams)), unmapped=" | ".join(unmapped),
                         single_name=len(names) == 1, single_material=len(bases) == 1 and not unmapped))
    return pd.DataFrame(rows)


def _context(quantities: pd.DataFrame, doc: str) -> dict:
    """Document-level thickness, temperature and RH. A single distinct value is reported as such; several are listed, not linked."""
    q = quantities[quantities["doc"] == doc]
    rh = q[q["target"].astype(str).str.contains("humid", case=False)]
    rh_vals = sorted({float(v) for v in rh["value"].dropna()} | {f"{lo:g}/{hi:g}" for lo, hi in rh[["low", "high"]].dropna().values}, key=str)
    out = {}
    for key, vals in (("thickness_um", sorted(q["thickness_um"].dropna().unique())), ("temperature_c", sorted(q["temperature_c"].dropna().unique())),
                      ("rh_pct", rh_vals)):
        if len(vals) == 1:
            out[key], out[f"{key}_context"] = vals[0], "document, single value"
        else:
            out[key] = np.nan
            out[f"{key}_context"] = ("document, several values not linked to rows: " + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)) if vals else "not reported"
    return out


def associate(df: pd.DataFrame, quantities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (materials, unresolved, per-document table)."""
    docs = doc_materials(df).set_index("doc")
    ok = quantities[quantities["unit_status"] == "ok"].copy()
    lo = ok["base_unit"].map(lambda u: PLAUSIBLE_RANGE[u][0])
    hi = ok["base_unit"].map(lambda u: PLAUSIBLE_RANGE[u][1])
    ok["plausible"] = ok["value_base"].between(lo, hi)
    mat, unres = [], []
    for r in ok.to_dict("records"):
        d = docs.loc[r["doc"]] if r["doc"] in docs.index else None
        reason = ""
        label = str(r.get("attached") or "").lower()
        if r["gas"] == "unspecified_gas":
            reason = "gas not named in the annotation"
        elif re.search(r"\b(otr|wvtr)\b", label):
            reason = f"annotation says a transmission rate ('{label}') but the unit is a permeability coefficient"
        elif not r["plausible"]:
            reason = f"implausible magnitude {r['value_base']:.3g} {r['base_unit']} (likely a dropped multiplier or wrong unit)"
        elif d is None:
            reason = "no packaging mention in the paper"
        elif not d["single_material"]:
            reason = (f"paper mentions {d['n_base_materials']} base materials ({d['base_materials']})"
                      + (f"; unmapped mentions: {d['unmapped']}" if d["unmapped"] else ""))
        base = dict(row_key=r["row_key"], source_doc=r["doc"], doi=r["doi"], gas=r["gas"], permeability=r["value_base"], unit=r["base_unit"],
                    permeability_low=None, permeability_high=None, value_as_reported=r["value"], unit_as_reported=r["unit"],
                    conversion_note=r["unit_note"], parse_note=r["parse_note"])
        if r["low"] == r["low"] and r["low"] is not None and r["value"]:
            base["permeability_low"] = r["value_base"] * r["low"] / r["value"]
            base["permeability_high"] = r["value_base"] * r["high"] / r["value"]
        if reason:
            unres.append(base | dict(reason=reason, paper_mentions=d["mentions"] if d is not None else ""))
            continue
        mat.append(dict(material=f"{d['base_materials']} film (variant not identified)", base_material=d["base_materials"],
                        family=d["families"], gas_or_vapour="vapour" if r["base_unit"] == VAPOUR_BASE else "gas") | base
                   | _context(quantities, r["doc"]) | dict(association="single-material paper", variant_resolved=False,
                                                            paper_mentions=d["mentions"]))
    return pd.DataFrame(mat), pd.DataFrame(unres), docs.reset_index()


def compare_with_placeholders(materials: pd.DataFrame, placeholder: pd.DataFrame) -> str:
    """Which placeholder rows now have a real value with a DOI. Nothing is replaced here."""
    fam_of = {"PE": "PE", "PP": "PP", "PET": "PET", "PA": "PA", "Paper": "paper", "EVOH": "EVOH", "Al": "Al"}
    lines = ["placeholder row -> real TRANSMAT values (single-material papers only; nothing replaced, no trust value changed)"]
    for _, p in placeholder.iterrows():
        fam = fam_of.get(p["family"], p["family"])
        hits = materials[materials["family"] == fam] if len(materials) else materials
        if p["barrier_type"] == "fixed" or not len(hits):
            why = "fixed-barrier layer (coating/foil/perforation): TRANSMAT has no matching values" if p["barrier_type"] == "fixed" else "no single-material paper for this family"
            lines.append(f"  {p['name']:<14} placeholder p_o2={p['p_o2']}, p_wv={p['p_wv']}: NO real value ({why})")
            continue
        parts = []
        for gas, g in hits.groupby("gas"):
            parts.append(f"{gas}: n={len(g)} range {g['permeability'].min():.3g}-{g['permeability'].max():.3g} {g['unit'].iloc[0]} "
                         f"({g['doi'].nunique()} DOI: {', '.join(sorted(g['doi'].unique()))})")
        lines.append(f"  {p['name']:<14} placeholder p_o2={p['p_o2']}, p_wv={p['p_wv']}: family {fam} has real values, but grade "
                     f"({p['name']}) and variant are not identified, so the row is NOT replaced. " + "; ".join(parts))
    return "\n".join(lines)

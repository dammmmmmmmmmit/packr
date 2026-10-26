"""Build the processed tables from the raw downloads and print an audit.

    python -m packr.build_tables --raw data/raw --out data/processed

Put these in the raw folder first (subfolders are searched):
    goldenTRANSMAT-all.tab     from https://dataverse.cirad.fr  (doi:10.18167/DVN1/U7HK8J); the comma-separated .csv also works
    Dataset.csv                from https://data.mendeley.com/datasets/czz68d9fwj
Outputs: transmat_quantities.csv, transmat_doc_summary.csv, materials_from_transmat.csv, transmat_unresolved.csv,
         transmat_doc_materials.csv, materials_comparison.txt, spoilage_by_fruit.csv, audit.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .associate import associate, compare_with_placeholders
from .materials import MATERIALS_CSV
from .paths import PROCESSED_DIR, RAW_DIR
from .spoilage import audit_spoilage, load_spoilage, summarise_by_fruit
from .transmat import audit_transmat, build_doc_summary, build_quantities, load_transmat


def association_report(df, quantities, out: Path) -> str:
    """Write materials_from_transmat.csv, transmat_unresolved.csv, transmat_doc_materials.csv, materials_comparison.txt."""
    materials, unresolved, docs = associate(df, quantities)
    materials.to_csv(out / "materials_from_transmat.csv", index=False)
    unresolved.to_csv(out / "transmat_unresolved.csv", index=False)
    docs.to_csv(out / "transmat_doc_materials.csv", index=False)
    comparison = compare_with_placeholders(materials, pd.read_csv(MATERIALS_CSV))
    (out / "materials_comparison.txt").write_text(comparison + "\n", encoding="utf-8")
    lines = ["MATERIAL ASSOCIATION (packr.associate)",
             f"papers naming exactly one packaging term: {int(docs['single_name'].sum())} of {len(docs)}",
             f"papers whose mentions all map to one base material: {int(docs['single_material'].sum())} of {len(docs)}",
             f"material rows written: {len(materials)} from {materials['doi'].nunique() if len(materials) else 0} DOIs "
             "(association = single-material paper; variant never resolved)",
             f"unresolved converted values: {len(unresolved)}"]
    if len(unresolved):
        why = unresolved["reason"].str.replace(r" \(.*", "", regex=True).str.replace(r"magnitude .*", "magnitude", regex=True)
        lines += [f"    {n:>4} x  {r}" for r, n in why.value_counts().items()]
    if len(materials):
        lines.append("material rows by base material and gas:")
        for (b, g), grp in materials.groupby(["base_material", "gas"]):
            lines.append(f"    {b:<24} {g:<13} n={len(grp):<3} {grp['permeability'].min():.3g} to {grp['permeability'].max():.3g} {grp['unit'].iloc[0]}")
    return "\n".join(lines + ["", comparison])


def _find(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.rglob(pattern)) if root.exists() else []
        if hits:
            return hits[0]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="packr.build_tables", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--transmat", type=Path, help="explicit path to goldenTRANSMAT-all.tab or .csv")
    ap.add_argument("--spoilage", type=Path, help="explicit path to the spoilage Dataset.csv")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    report: list[str] = []
    done = 0

    tm_path = args.transmat or _find(args.raw, ["goldenTRANSMAT-all.tab", "goldenTRANSMAT-all.csv", "*all*.tab"])
    if tm_path is None:
        report.append(f"TRANSMAT: goldenTRANSMAT-all.tab or .csv not found under {args.raw}")
    else:
        df = load_transmat(tm_path)
        quantities = build_quantities(df)
        quantities.to_csv(args.out / "transmat_quantities.csv", index=False)
        build_doc_summary(df, quantities).to_csv(args.out / "transmat_doc_summary.csv", index=False)
        report += [f"TRANSMAT file: {tm_path}", audit_transmat(df, quantities)]
        report += ["", association_report(df, quantities, args.out)]
        done += 1

    sp_path = args.spoilage or _find(args.raw, ["Dataset.csv", "dataset.csv"])
    if sp_path is None:
        report.append(f"SPOILAGE: Dataset.csv not found under {args.raw}")
    else:
        sp = load_spoilage(sp_path)
        summarise_by_fruit(sp).to_csv(args.out / "spoilage_by_fruit.csv", index=False)
        report += ["", f"Spoilage file: {sp_path}", audit_spoilage(sp)]
        done += 1

    text = "\n".join(report)
    (args.out / "audit.txt").write_text(text + "\n", encoding="utf-8")
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, errors="backslashreplace").decode(enc))     # Windows consoles cannot print the TRANSMAT minus sign
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

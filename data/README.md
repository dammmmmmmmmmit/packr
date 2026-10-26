# Data

`raw/` and `processed/` are git-ignored. Small curated tables live inside the package
(`src/packr/reference/`) so the engine works from a fresh install.

## Raw downloads (put them in `data/raw/`, subfolders are fine)

| File | Where from | Notes |
|---|---|---|
| `goldenTRANSMAT-all.tab` | https://dataverse.cirad.fr, doi:10.18167/DVN1/U7HK8J | Use the `-all` consensus file, not the per-annotator ones. 1,776 rows from 49 papers. |
| `Dataset.csv` | https://data.mendeley.com/datasets/czz68d9fwj | Fruit spoilage sensor data, 10,995 rows, 4 fruits. The site blocks scripted downloads: download by hand. |

## Reference tables (in the package)

| Table | Trust | What it is |
|---|---|---|
| `materials_placeholder.csv` | **placeholder** | 12 materials with indicative barrier values converted to permeability coefficients. Not from a paper or datasheet. Replace rows as real values arrive and change `trust`. |
| `commodities_produce_seed.csv` | **unverified_seed** | The 27 fresh-produce rows from the merge notebook (respiration, ideal T and RH, typical shelf life). Respiration values were compiled at differing, unrecorded temperatures. `resp_ref_temp_c`, `moisture_pct`, `fat_pct`, `ph` are empty and still to be filled. |

## Processed outputs (`python -m packr.build_tables`)

| File | One row per | Key columns |
|---|---|---|
| `transmat_quantities.csv` | quantity entity in TRANSMAT | `value`, `low`, `high`, `unit`, `gas`, `value_base`, `base_unit`, `unit_status` (`ok`, `unit_error`, `family_mismatch`, `no_value`), `unit_note`, `thickness_um`, `temperature_c` |
| `transmat_doc_summary.csv` | paper | `packaging_names`, `n_permeability_values`, `n_normalised`, `median_p_o2`, `median_p_co2`, `median_p_water_vapour`, `median_thickness_um`, temperature range |
| `spoilage_by_fruit.csv` | fruit | mean temperature, humidity, CO2, light, record count, share labelled good |
| `audit.txt` | run | counts, unit failures, whether the spoilage features predict the label at all |

`transmat_doc_summary.csv` is **document-level**: TRANSMAT does not say which value belongs to which
material inside a paper, so medians mix every material a paper tested. Do not use it as a per-material
barrier table.

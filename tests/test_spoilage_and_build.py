import numpy as np
import pandas as pd
import pytest

from packr.build_tables import main
from packr.spoilage import audit_spoilage, load_spoilage, summarise_by_fruit

COLUMNS = ["fruit", "temp", "humid (%)", "light (fux)", "co2 (pmm)", "class"]


def make_spoilage(n=600, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    fruit = rng.choice(["Banana", "Orange"], n)
    temp = rng.normal(24, 2, n)
    df = pd.DataFrame({"fruit": fruit, "temp": temp, "humid (%)": rng.normal(93, 2, n),
                       "light (fux)": rng.normal(15, 5, n), "co2 (pmm)": rng.normal(320, 20, n)})
    good = (temp < 24) if informative else rng.random(n) < 0.5
    df["class"] = np.where(good, "Good", "Bad")
    return df[COLUMNS]


@pytest.fixture
def spoilage_file(tmp_path):
    path = tmp_path / "Dataset.csv"
    make_spoilage().to_csv(path, index=False)
    return path


def test_load_renames_columns_and_normalises_labels(spoilage_file):
    df = load_spoilage(spoilage_file)
    assert {"fruit", "temp_c", "humidity_pct", "light_lux", "co2_ppm", "label"} <= set(df.columns)
    assert set(df["label"]) == {"good", "bad"}


def test_load_rejects_other_files(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="spoilage dataset"):
        load_spoilage(tmp_path / "x.csv")


def test_summary_matches_the_notebook_logic(spoilage_file):
    s = summarise_by_fruit(load_spoilage(spoilage_file)).set_index("commodity")
    raw = make_spoilage()
    assert s.loc["Banana", "n_records"] == (raw["fruit"] == "Banana").sum()
    assert s.loc["Banana", "avg_temp_c"] == pytest.approx(raw.loc[raw["fruit"] == "Banana", "temp"].mean())
    assert 0 <= s["pct_good"].min() and s["pct_good"].max() <= 100


def test_audit_tells_an_informative_dataset_from_noise(tmp_path):
    pytest.importorskip("sklearn")

    def auc(informative):
        p = tmp_path / f"d{informative}.csv"
        make_spoilage(informative=informative).to_csv(p, index=False)
        text = audit_spoilage(load_spoilage(p))
        assert "no packaging column present: True" in text
        return float(text.split("AUC of fruit + sensor features -> good/bad: ")[1].split()[0])

    assert auc(True) > 0.9
    assert 0.4 < auc(False) < 0.6


def test_build_tables_end_to_end(tmp_path, transmat_file, spoilage_file):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "nested").mkdir(parents=True)
    transmat_file.rename(raw / "goldenTRANSMAT-all.tab")
    spoilage_file.rename(raw / "nested" / "Dataset.csv")                # subfolders are searched
    assert main(["--raw", str(raw), "--out", str(out)]) == 0
    for name in ("transmat_quantities.csv", "transmat_doc_summary.csv", "spoilage_by_fruit.csv", "audit.txt"):
        assert (out / name).exists(), name
    assert "TRANSMAT: 12 rows" in (out / "audit.txt").read_text(encoding="utf-8")
    assert len(pd.read_csv(out / "transmat_doc_summary.csv")) == 2


def test_build_tables_reports_missing_inputs(tmp_path, capsys):
    assert main(["--raw", str(tmp_path / "empty"), "--out", str(tmp_path / "out")]) == 1
    assert "not found" in capsys.readouterr().out


def test_build_tables_finds_the_comma_separated_transmat(tmp_path, transmat_df):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    transmat_df.to_csv(raw / "goldenTRANSMAT-all.csv", index=False)
    assert main(["--raw", str(raw), "--out", str(out)]) == 0
    text = (out / "audit.txt").read_text(encoding="utf-8")
    assert "goldenTRANSMAT-all.csv" in text and "TRANSMAT: 12 rows" in text

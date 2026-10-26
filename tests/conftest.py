"""Fixtures shaped like the rows visible in the Packr merge notebook's TRANSMAT output.

The row formats (tuple repr, '???' for a lost en dash, '??' for a lost +/- / degree / micro sign) are taken
from that output. The unit tails after the first few tokens are ASSUMED, since the notebook printout was cut off.
"""
import pandas as pd
import pytest

DOI = "https://doi.org/10.1016/j.carbpol.2006.02.005"
DOC_A = "Barrier and surface properties of chitosan-coated films"
DOC_B = "Barrier properties of chitosan coated polyethylene"

PERM_CM3_MM = "['cm', '^', '3', 'mm', '/', '(', 'm', '^', '2', 'day', 'atm', ')']"
PERM_CM3_UM_KPA = "['cm', '^', '3', '??', 'm', '/', '(', 'm', '^', '2', 'day', 'kPa', ')']"
VAPOUR = "['g', 'mm', '/', '(', 'm', '^', '2', 'day', 'kPa', ')']"
PERMEANCE = "['cm', '^', '3', '/', '(', 'm', '^', '2', 'day', 'atm', ')']"


def row(i, doc, target, original, attached, typ, annotator=1):
    return dict(id=i, doc=doc, target=target, original_value=original, attached_value=attached,
                type=typ, annotator=annotator, doi=DOI)


@pytest.fixture
def transmat_df():
    rows = [
        row(0, DOC_A, "permeability", f"(['3400'], {PERM_CM3_MM})", "['carbon', 'dioxide']", "QUANTITY"),
        row(1, DOC_A, "o2_permeability", f"(['0.7', '???', '1.8'], {PERM_CM3_UM_KPA})", "['OP-values']", "QUANTITY", 3),
        row(2, DOC_A, "permeability", f"(['2'], {VAPOUR})", "['water', 'vapour']", "QUANTITY"),
        row(3, DOC_A, "o2_permeability", f"(['9'], {PERMEANCE})", "['OP-values']", "QUANTITY"),
        row(4, DOC_A, "impact_factor_component", "(['23', '??', '1'], ['??', 'C'])", "['temperature']", "QUANTITY"),
        row(5, DOC_A, "impact_factor_component", "(['50'], ['??', 'm'])", "['thickness']", "QUANTITY"),
        row(6, DOC_A, "partial_pressure_difference", "(['0'], ['%'])", None, "QUANTITY"),
        row(7, DOC_A, "permeability", "garbage that is not a tuple", "['oxygen']", "QUANTITY"),
        row(8, DOC_A, "packaging", "['chitosan', 'coated', 'PE']", None, "SYMBOLIC"),
        row(9, DOC_A, "packaging", "['Biopolymer']", None, "SYMBOLIC"),
        row(10, DOC_B, "packaging", "['PLA-coated', 'paperboard']", None, "SYMBOLIC"),
        row(11, DOC_B, "o2_permeability", f"(['4'], {PERM_CM3_MM})", "['OP-values']", "QUANTITY"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def transmat_file(tmp_path, transmat_df):
    path = tmp_path / "goldenTRANSMAT-all.tab"
    transmat_df.to_csv(path, sep="\t", index=False)
    return path

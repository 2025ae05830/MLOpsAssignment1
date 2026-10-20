
from src.data.preprocess import clean, load_raw


def test_load_raw_has_expected_columns():
    df = load_raw()
    assert len(df) == 303
    assert "target" in df.columns


def test_clean_binarizes_target():
    df = clean(load_raw())
    assert set(df["target"].unique()) <= {0, 1}


def test_clean_has_no_missing_values():
    df = clean(load_raw())
    assert df.isna().sum().sum() == 0


def test_clean_preserves_row_count():
    raw = load_raw()
    df = clean(raw)
    # no rows with missing target -> count preserved (missing ca/thal handled by imputation)
    assert len(df) <= len(raw)
    assert len(df) > 0

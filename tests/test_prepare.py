"""Smoke tests for the data-preparation step."""

from pathlib import Path

import pandas as pd

from src.prepare_data import PROJECT_ROOT, prepare


def test_prepare_adds_derived_columns(tmp_path: Path):
    out = prepare(PROJECT_ROOT / "data" / "raw" / "data.csv", tmp_path / "processed.csv")
    assert out.exists()

    df = pd.read_csv(out)
    assert {"grade_change", "grade_improved"}.issubset(df.columns)
    assert len(df) == 1000
    # grade_change is the rounded difference of the two exam scores
    assert (df["grade_change"] == round(df["final_exam_score"] - df["previous_grade"], 3)).all()
    assert df["grade_improved"].dtype == bool

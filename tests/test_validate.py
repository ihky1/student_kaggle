"""Smoke test for Great Expectations validation."""

from src.validate_data import PROJECT_ROOT, validate


def test_validate_processed_data_passes():
    assert validate(PROJECT_ROOT / "data" / "processed" / "data.csv") is True

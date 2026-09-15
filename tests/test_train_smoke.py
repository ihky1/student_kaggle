"""Smoke tests for the training module (no full training run here)."""

from src.prepare_data import PROJECT_ROOT
from src.train import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipelines,
    load_data,
    resolve_tracking_uri,
)


def test_load_data_drops_leakage_columns():
    X, y = load_data(PROJECT_ROOT / "data" / "processed" / "data.csv")
    assert TARGET_COLUMN not in X.columns
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(X.columns)
    assert len(X) == len(y) == 1000


def test_pipelines_fit_predict_on_sample():
    X, y = load_data(PROJECT_ROOT / "data" / "processed" / "data.csv")
    sample_X, sample_y = X.head(80), y.head(80)
    for name, pipeline in build_pipelines().items():
        pipeline.fit(sample_X, sample_y)
        preds = pipeline.predict(sample_X)
        assert len(preds) == len(sample_X)
        assert set(preds).issubset({True, False})


def test_tracking_uri_defaults_to_sqlite():
    assert resolve_tracking_uri().startswith("sqlite:///")

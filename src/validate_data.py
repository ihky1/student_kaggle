"""Validate the processed student dataset with Great Expectations.

Exposes ``validate(input_path) -> bool`` so the check can be reused from
other code and tests. As a script it exits with code 0 on success and 1 on
failure, so it can gate a pipeline::

    python src/validate_data.py && python src/train.py

The module has no side effects on import — everything runs inside ``main()``.
All GX objects are created with ``add_or_update_*`` / get-or-create fallbacks,
so the script is idempotent: running it twice in a row works.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import great_expectations as gx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUITE_NAME = "student_data_suite"
DATA_SOURCE_NAME = "pandas"
DATA_ASSET_NAME = "student_dataframe"
BATCH_DEFINITION_NAME = "student_batch"
VALIDATION_DEFINITION_NAME = "student_validation"

# (expectation_type, kwargs) pairs describing the processed dataset contract.
EXPECTATIONS: tuple = (
    ("ExpectColumnDistinctValuesToBeInSet", {"column": "gender", "value_set": ["Male", "Female"]}),
    (
        "ExpectColumnDistinctValuesToBeInSet",
        {"column": "internet_access", "value_set": ["Yes", "No"]},
    ),
    (
        "ExpectColumnDistinctValuesToBeInSet",
        {"column": "extracurricular_activities", "value_set": ["Yes", "No"]},
    ),
    (
        "ExpectColumnDistinctValuesToBeInSet",
        {"column": "part_time_job", "value_set": ["Yes", "No"]},
    ),
    (
        "ExpectColumnValuesToBeBetween",
        {"column": "attendance_percent", "min_value": 0.0, "max_value": 100.0},
    ),
    (
        "ExpectColumnDistinctValuesToBeInSet",
        {
            "column": "parental_education",
            "value_set": ["High School", "Bachelors", "Masters", "PhD"],
        },
    ),
    (
        "ExpectColumnValuesToBeBetween",
        {"column": "previous_grade", "min_value": 0.0, "max_value": 100.0},
    ),
    (
        "ExpectColumnValuesToBeBetween",
        {"column": "final_exam_score", "min_value": 0.0, "max_value": 100.0},
    ),
    (
        "ExpectColumnDistinctValuesToBeInSet",
        {"column": "final_grade", "value_set": ["A", "B", "C", "D", "F"]},
    ),
)


def _get_or_add_asset(data_source, name: str):
    """Return the dataframe asset, creating it if it does not exist yet."""
    try:
        return data_source.add_dataframe_asset(name=name)
    except Exception:
        return data_source.get_asset(name)


def _get_or_add_batch_definition(asset, name: str):
    """Return the whole-dataframe batch definition, creating it if needed."""
    try:
        return asset.add_batch_definition_whole_dataframe(name)
    except Exception:
        return asset.get_batch_definition(name)


def build_suite(context) -> object:
    """Create (or reuse) the expectation suite describing valid student data."""
    suite = context.suites.add_or_update(
        gx.core.expectation_suite.ExpectationSuite(name=SUITE_NAME)
    )
    for expectation_type, kwargs in EXPECTATIONS:
        suite.add_expectation(getattr(gx.expectations, expectation_type)(**kwargs))
    return suite


def validate(input_path: Path) -> bool:
    """Validate the CSV at ``input_path``. Returns True when all checks pass."""
    df = pd.read_csv(input_path)
    context = gx.get_context()

    data_source = context.data_sources.add_or_update_pandas(DATA_SOURCE_NAME)
    asset = _get_or_add_asset(data_source, DATA_ASSET_NAME)
    batch_definition = _get_or_add_batch_definition(asset, BATCH_DEFINITION_NAME)
    suite = build_suite(context)

    validation_definition = context.validation_definitions.add_or_update(
        gx.core.validation_definition.ValidationDefinition(
            name=VALIDATION_DEFINITION_NAME,
            data=batch_definition,
            suite=suite,
        )
    )
    result = validation_definition.run(batch_parameters={"dataframe": df})
    print(f"Validation successful: {result.success}")
    return bool(result.success)


def main() -> None:
    parser = ArgumentParser(description="Validate the processed student dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "data.csv",
        help="Path to the processed CSV.",
    )
    args = parser.parse_args()
    sys.exit(0 if validate(args.input) else 1)


if __name__ == "__main__":
    main()

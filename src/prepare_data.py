"""Prepare raw student data for modelling.

Reads ``data/raw/data.csv``, adds two derived columns and writes
``data/processed/data.csv``:

- ``grade_change``: ``final_exam_score - previous_grade`` (rounded to 3 decimals)
- ``grade_improved``: ``True`` when the final score is higher than the previous grade

The module has no side effects on import — everything runs inside ``main()``,
so other modules (and tests) can safely do
``from src.prepare_data import prepare``.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare(input_path: Path, output_path: Path) -> Path:
    """Build the processed dataset and return the output path."""
    df = pd.read_csv(input_path)

    df["grade_change"] = round(df["final_exam_score"] - df["previous_grade"], 3)
    df["grade_improved"] = df["grade_change"] > 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"wrote {output_path} shape={df.shape}")
    return output_path


def main() -> None:
    parser = ArgumentParser(description="Prepare raw student data for modelling.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "data.csv",
        help="Path to the raw Kaggle CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "data.csv",
        help="Where to write the processed CSV.",
    )
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()

"""Train student grade-improvement classifiers with MLflow tracking.

Target: ``grade_improved`` (True when ``final_exam_score > previous_grade``,
built by ``src/prepare_data.py``).

Leakage control: ``final_exam_score``, ``final_grade`` and ``grade_change``
are known only AFTER the exam (or derived from the target), so they — together
with the ``student_id`` identifier — are dropped from the features.

Pipeline: compare LogisticRegression / RandomForest / SVM with 5-fold CV on
the train split, then tune the winner (logistic regression, ``C`` grid) and
evaluate it once on the held-out test set. The tuned model is saved to
``model/artifacts/model.pkl`` and logged to MLflow together with test metrics.

Usage (from the repo root)::

    python src/train.py
    python src/train.py --test-size 0.25 --random-state 7

The module has no side effects on import — everything runs inside ``main()``.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Experiment config: all former "magic numbers" live here, in one place. ---
EXPERIMENT_NAME = "student_grade_improvement_prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = ["accuracy", "precision", "recall", "f1"]
PARAM_GRID = {"classifier__C": [0.01, 0.1, 1, 10, 100]}

TARGET_COLUMN = "grade_improved"
LEAKAGE_COLUMNS = ["grade_change", "final_grade", "final_exam_score", "student_id"]
NUMERIC_FEATURES = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]
CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
]


def resolve_tracking_uri() -> str:
    """MLflow store location. Env var wins, otherwise sqlite DB in ``model/``.

    MLflow 3.x put the plain file store (``./mlruns``) into maintenance mode,
    so the default is a local sqlite backend — it keeps working without any
    setup and stays inside ``model/`` (git-ignored).
    """
    default = f"sqlite:///{PROJECT_ROOT / 'model' / 'mlflow.db'}"
    return os.environ.get("MLFLOW_TRACKING_URI", default)


def load_data(data_path: Path):
    """Load the processed CSV and split it into leakage-free ``X`` and target ``y``."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS])
    y = df[TARGET_COLUMN]
    return X, y


def build_pipelines(random_state: int = RANDOM_STATE) -> dict:
    """Build the three candidate ``{name: Pipeline}`` classifiers."""
    numeric_logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    logreg_preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_logreg, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )

    numeric_rf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    rf_preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_rf, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )

    return {
        "logistic_regression": Pipeline(
            [("preprocessor", logreg_preprocessor), ("classifier", LogisticRegression())]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", rf_preprocessor),
                ("classifier", RandomForestClassifier(random_state=random_state)),
            ]
        ),
        "svm": Pipeline([("preprocessor", logreg_preprocessor), ("classifier", SVC())]),
    }


def tune_and_evaluate(
    pipeline: Pipeline, X_train, X_test, y_train, y_test, cv_folds: int, output_path: Path
) -> Pipeline:
    """Tune logistic regression, log test metrics + model artifact, save model.

    The model is serialized with joblib to ``output_path`` and that same file
    is logged to MLflow via ``log_artifact`` — one file, two homes (disk for
    inference, MLflow for experiment history).
    """
    search = GridSearchCV(pipeline, PARAM_GRID, cv=cv_folds, scoring="f1")
    with mlflow.start_run(run_name="logistic_regression_tuned"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("param_grid", str(PARAM_GRID))

        # Note: a single fit is enough — GridSearchCV refits the best estimator
        # on the full train set by default (refit=True).
        search.fit(X_train, y_train)
        mlflow.log_param("best_C", search.best_params_["classifier__C"])
        mlflow.log_metric("best_cv_f1", search.best_score_)

        final_model = search.best_estimator_
        y_pred = final_model.predict(X_test)

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(test_metrics)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="model")

    print(f"Best C: {search.best_params_['classifier__C']}")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return final_model


def main() -> None:
    parser = ArgumentParser(description="Train grade-improvement classifiers.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "data.csv",
        help="Path to the processed CSV.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "model.pkl",
        help="Where to save the tuned model.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(resolve_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipelines = build_pipelines(random_state=args.random_state)
    for run_name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", type(pipeline.named_steps["classifier"]).__name__)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            scores = cross_validate(pipeline, X_train, y_train, cv=CV_FOLDS, scoring=SCORING)
            for metric in SCORING:
                mlflow.log_metric(f"cv_{metric}_mean", scores[f"test_{metric}"].mean())

    tune_and_evaluate(
        pipelines["logistic_regression"],
        X_train,
        X_test,
        y_train,
        y_test,
        CV_FOLDS,
        args.output,
    )
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()

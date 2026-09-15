# Student Grade Improvement Prediction

Predict whether a student's final exam score will improve over their previous
grade — a binary classification task on tabular data with a full ML pipeline:
EDA → data preparation → Great Expectations validation → model comparison with
MLflow tracking → tuned model artifact.

## Problem & target

- **Target:** `grade_improved = final_exam_score > previous_grade` (built in `src/prepare_data.py`)
- **Why it matters:** early identification of students at risk of slipping lets
  teachers intervene before the final exam.
- **Leakage control:** `final_exam_score`, `final_grade` and `grade_change` are
  known only *after* the exam, so they are dropped from the features together
  with the `student_id` identifier (see `LEAKAGE_COLUMNS` in `src/train.py`).
- **Imbalance note:** ~86% of students improved, so accuracy alone is
  misleading — models are compared on **F1** (5-fold CV) and the final model is
  evaluated once on a held-out stratified test split.

## Data

Source: [Student Performance & Study Habits Dataset (Kaggle)](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset)
— 1000 rows × 12 columns. Raw CSVs are **not** committed; download `data.csv`
from Kaggle and place it at `data/raw/data.csv` (details in `data/README.md`).

## Project structure

```
├── data/
│   ├── raw/          # data.csv from Kaggle (git-ignored, see data/README.md)
│   └── processed/    # built by src/prepare_data.py (git-ignored)
├── notebooks/
│   └── eda.ipynb     # exploratory analysis + modelling conclusions
├── src/
│   ├── prepare_data.py   # raw → processed (CLI: --input/--output)
│   ├── validate_data.py  # Great Expectations check, exit 0/1
│   └── train.py          # CV comparison + tuning + test eval (CLI)
├── model/
│   ├── artifacts/    # model.pkl (regenerated, git-ignored)
│   └── mlflow.db     # local MLflow backend (git-ignored)
├── tests/            # pytest smoke tests
├── Makefile          # setup / data / validate / train / test / lint
└── requirements.txt
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put data.csv from Kaggle into data/raw/data.csv, then:
make pipeline   # prepare → validate → train
make test       # pytest smoke tests
make lint       # ruff
```

Or step by step:

```bash
.venv/bin/python src/prepare_data.py
.venv/bin/python src/validate_data.py && echo VALIDATION-OK
.venv/bin/python src/train.py            # try --help for options
```

MLflow UI: `MLFLOW_TRACKING_URI=sqlite:///model/mlflow.db .venv/bin/mlflow ui`

## Results

5-fold CV on the train split (`test_size=0.2`, `random_state=42`, stratified):

| model              | cv_accuracy | cv_precision | cv_recall | cv_f1  |
| ------------------ | ----------: | -----------: | --------: | -----: |
| LogisticRegression |      0.9100 |       0.9322 |    0.9668 | 0.9490 |
| RandomForest       |      0.8988 |       0.9080 |    0.9827 | 0.9438 |
| SVM                |      0.9013 |       0.9162 |    0.9755 | 0.9448 |

Logistic regression won on average, so it was tuned (`C ∈ {0.01, 0.1, 1, 10, 100}`).
Best `C=1`, held-out test performance:

| test_accuracy | test_precision | test_recall | test_f1 |
| ------------: | -------------: | ----------: | ------: |
|        0.9250 |         0.9341 |      0.9827 |  0.9577 |

The tuned pipeline is saved to `model/artifacts/model.pkl` (joblib) and logged
to MLflow with test metrics.

## Limitations & next steps

- Target is heavily imbalanced (86% positive) — worth trying class weights,
  threshold tuning, or PR-AUC as the selection metric.
- Single 80/20 split for final eval — nested CV would give a sturdier estimate.
- No inference service yet — the saved `model.pkl` is ready for a thin
  FastAPI `/predict` wrapper as a follow-up.
- EDA conclusions live at the bottom of `notebooks/eda.ipynb`.

## Author

[ihky1](https://github.com/ihky1) — Junior ML Engineer portfolio project (MIT).

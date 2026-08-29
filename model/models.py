import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_experiment("student_grade_improvement_prediction")

df = pd.read_csv("./data/processed/data.csv")

X = df.drop(columns=["grade_improved", "grade_change", "final_grade", "final_exam_score", "student_id"])
y = df["grade_improved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade']
categorical_features = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job']

## LOGISTICAL REGRESSION CLASSIFIER :

logreg_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

logreg_preprocessor = ColumnTransformer([
    ('numeric', logreg_numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

logreg_pipeline = Pipeline([
    ('preprocessor', logreg_preprocessor),
    ('classifier', LogisticRegression()),
])

with mlflow.start_run(run_name='logistic_regression'):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    mlflow.log_param("model", "LogisticRegression")

    metrics = cross_validate(logreg_pipeline, X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    for metric in scoring:
            mlflow.log_metric(f'cv_{metric}_mean', metrics[f'test_{metric}'].mean())

## RANDOM FOREST CLASSIFIER :

rf_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

rf_preprocessor = ColumnTransformer([
    ('numeric', rf_numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

random_forest_pipeline = Pipeline([
    ('preprocessor', rf_preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

with mlflow.start_run(run_name='random_forest'):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    mlflow.log_param("model", "RandomForestClassifier")

    metrics = cross_validate(random_forest_pipeline, X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    for metric in scoring:
        mlflow.log_metric(f'cv_{metric}_mean', metrics[f'test_{metric}'].mean())
    
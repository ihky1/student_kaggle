import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    
## SVM CLASSIFIER :

svm_pipeline = Pipeline([
    ('preprocessor', logreg_preprocessor),
    ('classifier', SVC())
])

with mlflow.start_run(run_name='svm'):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    mlflow.log_param("model", "SVC")

    metrics = cross_validate(svm_pipeline, X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
    for metric in scoring:
        mlflow.log_metric(f'cv_{metric}_mean', metrics[f'test_{metric}'].mean())


## Logistic Regression model performs better on average!

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    logreg_pipeline,
    param_grid,
    cv=5,
    scoring='f1'
)

with mlflow.start_run(run_name='logistic_regression_tuned'):
    mlflow.log_param("model", "LogisticRegression")

    grid_search.fit(X_train, y_train)
    mlflow.log_param(
        "best_C",
        grid_search.best_params_["classifier__C"]
    )

    mlflow.log_metric(
        "best_cv_f1",
        grid_search.best_score_
    )


final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
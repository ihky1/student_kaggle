import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
import mlflow

mlflow.set_experiment("student_grade_improvement_prediction")

df = pd.read_csv("./data/processed/data.csv")

X = df.drop(columns=["grade_improved", "grade_change", "final_grade", "final_exam_score", "student_id"])
y = df["grade_improved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade']
categorical_features = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
])

with mlflow.start_run():
    logreg_pipeline.fit(X_train, y_train)
    score = cross_validate(logreg_pipeline, X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])

    mlflow.log_metric('cv_accuracy_mean', score['test_accuracy'].mean())
    mlflow.log_metric('cv_precision_mean', score['test_precision'].mean())
    mlflow.log_metric('cv_recall_mean', score['test_recall'].mean())
    mlflow.log_metric('cv_f1_mean', score['test_f1'].mean())


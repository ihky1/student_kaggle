import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import mlflow


df = pd.read_csv("../data/processed/data.csv")

X = df.drop(columns=["grade_improved", "grade_change", "final_grade", "final_exam_score"])
y = df["grade_improved"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
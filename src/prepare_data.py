import pandas as pd


df = pd.read_csv("../data/raw/data.csv")

df['grade_change'] = round(df['final_exam_score'] - df['previous_grade'], 3)
df['grade_improved'] = df['grade_change'] > 0

# SOME EXPECTATIONS





df.to_csv("../data/processed/data.csv", index=False)
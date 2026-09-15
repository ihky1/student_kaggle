# Data

Source: Kaggle - Student Performance & Study Habits Dataset: https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset

- Download `data.csv` from Kaggle and put it as `data/raw/data.csv`
- Then run: `python src/prepare_data.py`
- Output: `data/processed/data.csv` with 2 extra columns: `grade_change`, `grade_improved`

Raw shape: 1000 x 12, target for model: `grade_improved = final_exam_score > previous_grade`
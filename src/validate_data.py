import great_expectations as gx
import pandas as pd

df = pd.read_csv("../data/processed/data.csv")

context = gx.get_context()

data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")

suite = context.suites.add(
    gx.core.expectation_suite.ExpectationSuite(name="student_data_suite")
)

suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="gender", value_set=["Male", "Female"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="internet_access", value_set=["Yes", "No"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="extracurricular_activities", value_set=["Yes", "No"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="part_time_job", value_set=["Yes", "No"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="attendance_percent", min_value=0.0, max_value=100.0)
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="parental_education", value_set=["High School", "Bachelors", "Masters", "PhD"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="previous_grade", min_value=0.0, max_value=100.0)
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="final_exam_score", min_value=0.0, max_value=100.0)
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="final_grade", value_set=["A", "B", "C", "D", "F"])
)

validation_definition = context.validation_definitions.add(
    gx.core.validation_definition.ValidationDefinition(
        name="validation definition",
        data=batch_definition,
        suite=suite,
    )
)

result = validation_definition.run(
    batch_parameters={"dataframe": df}
)

print("Validation successful: {}".format(result.success))
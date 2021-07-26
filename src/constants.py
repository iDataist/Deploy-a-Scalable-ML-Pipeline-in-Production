data_path = "data/census.csv"
processed_data_path = "data/processed_data.csv"
eda_output_path = "results/eda/eda.html"
image_output_path = "results/images/"
model_output_path = "models/rfc_model.pkl"
log_path = "logs/train_model.log"
cat_columns = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "salary",
]
quant_columns = [
    "age",
    "fnlgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
target = "salary_>50K"
random_state = 0
test_size = 0.3

data_path = '../data/census.csv'
eda_output_path = '../results/eda/eda.html'
image_output_path = '../results/images/'
model_output_path = '../models/rfc_model.pkl'
log_path = '../logs/train_model.log'
cat_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'salary'
]
quant_columns = [
    'age',
    'fnlgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week']
target = 'salary_>50K'
random_state = 0
test_size = 0.3

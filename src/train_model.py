from ml.data import (
    import_data,
    perform_eda,
    scaler,
    encoder,
    perform_train_test_split,
)
from ml.model import train_models, inference, compute_model_metrics
import constants
import logging

logging.basicConfig(
    level=logging.INFO,
    filename=constants.log_path,
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()

logger.info("############################################################")
logger.info("import data")
df = import_data(constants.data_path)
logger.info(f"inspect dataframe:     \n{df.iloc[0]}")
logger.info(f"generate EDA report: {constants.eda_output_path}")
perform_eda(df, constants.eda_output_path)
logger.info(f"normalize numeric features: {constants.quant_columns}")
df = scaler(df, constants.quant_columns)
logger.info(f"inspect dataframe:     \n{df.iloc[0]}")
logger.info(f"one-hot-encode categorical features:{constants.cat_columns}")
df = encoder(df, constants.cat_columns)
logger.info(f"inspect dataframe:     \n{df.iloc[0]}")
df.to_csv(constants.processed_data_path, index=False)
logger.info(
    f"perform train test split with the test size of {constants.test_size}"
)
X_train, X_test, y_train, y_test = perform_train_test_split(
    df, constants.target, constants.test_size, constants.random_state
)
logger.info("start training")
best_model, best_params = train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    constants.image_output_path,
    constants.model_output_path,
)
logger.info(f"best parameters are: {best_params}")
logger.info(
    f"save models in {constants.model_output_path}, "
    + f"store results in {constants.image_output_path}"
)

preds = inference(best_model, X_test)
precision, recall, f1 = compute_model_metrics(y_test, preds)
logger.info(f"precision: {precision}, recall: {recall}, f1: {f1}")

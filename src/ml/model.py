# from sklearn.metrics import fbeta_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np

# # Optional: implement hyperparameter tuning.
# def train_model(X_train, y_train):
#     """
#     Trains a machine learning model and returns it.

#     Inputs
#     ------
#     X_train : np.array
#         Training data.
#     y_train : np.array
#         Labels.
#     Returns
#     -------
#     model
#         Trained machine learning model.
#     """

#     # Number of trees in random forest
#     n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in np.linspace(2, 100, num = 5)]
#     # Minimum number of samples required to split a node
#     min_samples_split = [2, 5, 10]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2, 4]

#     # Create the random grid
#     random_grid = {'n_estimators': n_estimators,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf}
#     # Use the random grid to search for best hyperparameters
#     # First create the base model to tune
#     rf = RandomForestClassifier()
#     # Random search of parameters, using 3 fold cross validation,
#     # search across 10 different combinations, and use all available cores
#     rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#     # Fit the random search model
#     rf_random.fit(X_train, y_train)
# #     return rf_random.best_estimator_

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score, precision_score, recall_score



def classification_report_image(
    y_train, y_test, y_train_preds, y_test_preds, output_path
):

    """
    produces classification report for training and testing results
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            output_path: path to store the figure
    output:
            None
    """

    plt.rc("figure", figsize=(7, 5))
    plt.text(
        0.01, 1.1, str("Train"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01, 0.5, str("Test"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.1,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(output_path + "classification_report.png")
    plt.close()


def feature_importance_plot(model, X, output_path):
    """
    creates and stores the feature importances in output_path
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]
    plt.figure(figsize=(20, 20))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=60)
    plt.savefig(output_path + "feature_importance.png")
    plt.close()


def train_models(
    X_train, X_test, y_train, y_test, image_output_path, model_output_path
):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              image_output_path: path to store the figures
              model_output_path: path to store the models
    output:
              best_model
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 5)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 40, num = 5)]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 15, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 8, 16]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 10 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_model = rf_random.best_estimator_
    best_params = rf_random.best_params_
    # creates and stores the feature importances
    feature_importance_plot(best_model, X_test, image_output_path)
    # produces classification report for training and testing results
    y_train_preds = best_model.predict(X_train)
    y_test_preds = best_model.predict(X_test)
    classification_report_image(
        y_train, y_test, y_train_preds, y_test_preds, image_output_path
    )
    # saves best model
    joblib.dump(best_model, model_output_path)

    return best_model, best_params

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """
    f1 = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, f1


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

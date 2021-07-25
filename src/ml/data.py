import pandas as pd
import pandas_profiling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def import_data(path):
    """
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(path)
    return df


def perform_eda(df, output_path):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_path: path to store the eda report

    output:
            None
    """
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(output_path)


def scaler(df, quant_columns):
    """
    helper function to normalize each numerical column
    input:
            df: pandas dataframe
    output:
            df: normalized pandas dataframe
    """
    df[quant_columns] = StandardScaler().fit_transform(df[quant_columns])
    return df


def encoder(df, cat_columns):
    """
    helper function to one-hot-encode each categorical column
    input:
            df: pandas dataframe
    output:
            df: one-hot-encoded pandas dataframe
    """
    return pd.get_dummies(df, columns=cat_columns, drop_first=True)


def perform_train_test_split(df, target, test_size, random_state):
    """
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              target: target column
    """
    X = df.drop(columns=[target])
    y = df[target].ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
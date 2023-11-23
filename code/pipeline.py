import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import scipy.stats as stats
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from typing import Tuple
from tqdm.auto import tqdm
from sklearn.base import clone


def load_library(path: str) -> pd.DataFrame:
    """Load the dataset

    Args:
        path (str): Path to the dataset

    Returns:
        pd.DataFrame: Dataframe with the dataset
    """
    df = pd.read_csv(path)
    return df


def del_duplicated(df: pd.DataFrame) -> pd.DataFrame:
    """Delete duplicated rows

    Args:
        df (Pandas Dataframe): Dataframe to be cleaned

    Returns:
        Pandas Dataframe: Dataframe without duplicated rows
    """
    df.drop_duplicates(inplace=True)
    return df


def time_to_hours(df: pd.DataFrame):
    """Convert 'Time' to hours

    Args:
        df (Pandas Dataframe): Dataframe to be cleaned

    Returns:
        Pandas Dataframe: Dataframe with 'Time' converted to hours
    """
    df["hour"] = df["Time"].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    df["hour48"] = df["Time"].apply(lambda x: np.ceil(float(x) / 3600) % 48)

    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Split the data into training and testing sets but keep the same class distribution as the original dataset

    Args:
        df (pd.DataFrame): Dataframe to be split

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training and testing sets
    """
    # Order the dataset by time
    df = df.sort_values(by=["Time"])

    # Create the X and y datasets
    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    # Divide two days. Train with the first day and test with the second day
    # divide indices into two groups: hour48 < 24 and hour48 >= 24
    # train with the first group and test with the second group
    X_train = X[X["hour48"] < 24]
    X_test = X[X["hour48"] >= 24]
    y_train = y[X["hour48"] < 24]
    y_test = y[X["hour48"] >= 24]

    X_train = X_train.drop(["hour48", "Time"], axis=1)
    X_test = X_test.drop(["hour48", "Time"], axis=1)

    # Divide the training set into train and validation sets (85% and 15%) and keep the same class distribution
    X_train_85, X_val_15, y_train_85, y_val_15 = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    return X_train, X_test, y_train, y_test, X_train_85, X_val_15, y_train_85, y_val_15


def evaluate_ml_models(models, X_train, y_train, X_val, y_val):
    """Evaluate the models using a training and validation set.
    Args:
        models (list): List of (model, name) tuples.
        X_train (DataFrame): Training data features.
        y_train (Series): Training data labels.
        X_val (DataFrame): Validation data features.
        y_val (Series): Validation data labels.
    Returns:
        dict: A dictionary with model performance summary.
    """
    model_summary = {}
    for model, name in models:
        print(f"Training {name}...")

        # Fit the model with a progress bar (specific to RandomForest)
        if name == "Random Forest":
            model = fit_with_progress_bar(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_pred_prob)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)

        # Add model performance to summary
        model_summary[name] = {
            "ROC AUC": auc_score,
            "F1 Score": f1,
            "Accuracy": accuracy,
        }

    return model_summary


def fit_with_progress_bar(model, X, y):
    """Fit a RandomForest model with a progress bar."""
    model_clone = clone(model)
    n_estimators = model_clone.get_params()["n_estimators"]

    with tqdm(total=n_estimators, desc="Training Progress") as pbar:
        for i in range(1, n_estimators + 1):
            model_clone.set_params(n_estimators=i)
            model_clone.fit(X, y)
            pbar.update(1)

    return model_clone


def train_nn(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train a neural network

    Args:
        X_train (pd.DataFrame): Training set
        y_train (pd.DataFrame): Training labels

    Returns:
        Tuple[Sequential, History]: Trained model and training history
    """
    model_nn = Sequential()
    model_nn.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(64, activation="relu"))
    model_nn.add(Dense(1, activation="sigmoid"))
    model_nn.compile(
        loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
    )
    callback = EarlyStopping(monitor="val_loss", patience=5)
    history = model_nn.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=128,
        callbacks=[callback],
    )

    y_pred_nn = (model_nn.predict(X_test) > 0.5).astype(int).ravel()
    auc_nn = roc_auc_score(y_test, y_pred_nn)
    f1_nn = f1_score(y_test, y_pred_nn)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)

    # Add neural network performance to summary
    model_summary["Neural Network"] = {
        "ROC AUC": auc_nn,
        "F1 Score": f1_nn,
        "Accuracy": accuracy_nn,
    }

    return model_nn, history, model_summary


if __name__ == "__main__":
    # Load the dataset
    df = load_library("code/creditcard.csv")

    # Clean the dataset
    df_modified = del_duplicated(df)
    df_modified = time_to_hours(df_modified)

    # Split the dataset
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_85,
        X_val_15,
        y_train_85,
        y_val_15,
    ) = split_data(df_modified)

    # TODO - RANDOM SEARCH FOR HYPERPARAMETER TUNING
    # Define a list of (model, name) tuples
    models = [
        (RandomForestClassifier(n_estimators=2, random_state=42), "Random Forest"),
        (SVC(probability=True), "Support Vector Machine"),
    ]

    # Dictionary to hold summary results
    model_summary = {}

    # Evaluate the models using cross-validation
    model_summary = evaluate_ml_models(
        models, X_train_85, y_train_85, X_val_15, y_val_15
    )

    print(model_summary)

    # TODO Guardar modelo, e guardar imagens de performance

    # Train a neural network
    # model_nn, history, model_summary = train_nn(X_train, y_train)

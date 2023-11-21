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
    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets but keep the same class distribution as the original dataset

    Args:
        df (pd.DataFrame): Dataframe to be split

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
    """
    # Create the X and y datasets
    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    # Split the data into training and testing sets but keep the same class distribution as the original dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
    )

    return X_train, X_test, y_train, y_test


def evaluate_cv_models(models: list, model_summary: dict):
    """Evaluate the models using cross-validation
    Args:
        models (list): List of (model, name) tuples
    """
    for model, name in models:
        cv_auc_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Add model performance to summary
        model_summary[name] = {
            "ROC AUC": auc_score,
            "F1 Score": f1,
            "Accuracy": accuracy,
        }
    return model_summary


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
    df = load_library("creditcard.csv")

    # Clean the dataset
    df_modified = del_duplicated(df)
    df_modified = time_to_hours(df_modified)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(df_modified)

    # Define a list of (model, name) tuples
    models = [
        (SVC(), "Support Vector Machine"),
        (RandomForestClassifier(n_estimators=30, random_state=42), "Random Forest"),
    ]

    # Dictionary to hold summary results
    model_summary = {}

    # Evaluate the models using cross-validation
    model_summary = evaluate_cv_models(models, model_summary)

    # Train a neural network
    model_nn, history, model_summary = train_nn(X_train, y_train)

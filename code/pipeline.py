import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import scipy.stats as stats
from sklearn.svm import SVC, OneClassSVM
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
import os
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump

import warnings

# Ignore specific UserWarnings from SciPy about NumPy version
warnings.filterwarnings("ignore")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified path.
    Args:
        path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows from the dataframe.
    Args:
        df (pd.DataFrame): Dataframe to clean.
    Returns:
        pd.DataFrame: Dataframe without duplicates.
    """
    df = df.drop_duplicates(inplace=True)
    return df


def add_hour_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour columns to the dataframe based on 'Time'.
    Args:
        df (pd.DataFrame): Dataframe to process.
    Returns:
        pd.DataFrame: Dataframe with added hour columns.
    """
    df["hour"] = df["Time"].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    df["hour48"] = df["Time"].apply(lambda x: np.ceil(float(x) / 3600) % 48)

    return df


# TODO - Time-Based Cross-Validation: Instead of a single 85%-15% split for training-validation, consider using time-based cross-validation. This involves using several train-validation splits over time. For example, you might use the first 70% of day 1 for training and the next 15% for validation, then shift this window forward in time for the next fold.
def create_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature and label sets from the dataframe.
    Args:
        df (pd.DataFrame): Dataframe to process.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and labels (y).
    """
    df = df.sort_values(by=["Time"])
    X = df.drop(["Class"], axis=1)
    y = df["Class"]
    return X, y


def split_into_train_test(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets based on 'hour48'.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Label set.
    Returns:
        Tuple: Training and testing feature and label sets.
    """
    X_train = X[X["hour48"] < 24].drop(["hour48", "Time"], axis=1)
    y_train = y[X["hour48"] < 24]
    X_test = X[X["hour48"] >= 24].drop(["hour48", "Time"], axis=1)
    y_test = y[X["hour48"] >= 24]

    X_train = X_train.drop(["hour48", "Time"], axis=1)
    X_test = X_test.drop(["hour48", "Time"], axis=1)
    return X_train, X_test, y_train, y_test


def train_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split training data into training and validation sets.
    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training label set.
        test_size (float): Proportion of validation set.
        random_state (int): Seed for random number generator.
    Returns:
        Tuple: Training and validation feature and label sets.
    """
    return train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )


def save_model(model: object, model_name: str, directory: str = "bestmodels/"):
    # Add model name to directory
    directory += model_name + "/"

    # Create the directory if it doesn't exist
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created" % directory)
    else:
        pass

    # Save the model
    try:
        dump(model, directory + model_name + ".joblib")
    except OSError as error:
        print("Model '%s' can not be saved" % model_name)
    else:
        print("Successfully saved the model '%s'" % model_name)


def evaluate_ml_models(
    models, params, X_train, y_train, X_val, y_val, n_iter=100, cv=5
):
    """Evaluate the models using a training and validation set.
    Args:
        models (list): List of (model, name) tuples.
        X_train (DataFrame): Training data features.
        y_train (Series): Training data labels.
        X_val (DataFrame): Validation data features.
        y_val (Series): Validation data labels.
    Returns:
        dict, dict: Summary and predictions for each model.
    """

    model_summary_evaluation = {}
    model_predictions_evaluation = {}

    for model, name in models:
        print(f"Evaluating {name}...")

        # Random search for hyperparameter tuning
        random_search = RandomizedSearchCV(
            model,
            params[name],
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        y_pred = best_model.predict(X_val)
        y_pred_prob = best_model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_pred_prob)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)

        # Add model performance to summary
        model_summary_evaluation[name] = {
            "ROC AUC": auc_score,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Best Params": random_search.best_params_,
        }

        model_predictions_evaluation[name] = {
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
        }

    return model_summary_evaluation, model_predictions_evaluation


def train_ml_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_summary_evaluation: dict,
) -> Tuple[dict, dict]:
    """Train the models on the entire training set

    Args:
        X_train (pd.DataFrame): Training set
        y_train (pd.DataFrame): Training labels
        X_test (pd.DataFrame): Testing set
        y_test (pd.DataFrame): Testing labels
        model_summary_evaluation (dict): Summary of the models

    Returns:
        Tuple[dict, dict]: Summary and predictions for each model
    """

    for model, name in models:
        print(f"Training {name}...")

        # define the model with the best parameters
        model = model.set_params(**model_summary_evaluation[name]["Best Params"])

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Add model performance to summary
        model_summary[name] = {
            "ROC AUC": auc_score,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Best Params": model_summary_evaluation[name]["Best Params"],
        }

        model_predictions[name] = {"y_pred": y_pred, "y_pred_prob": y_pred_prob}

        save_model(model, name)

    return model_summary, model_predictions


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


def save_confusion_matrix(y_true, y_pred, model_name, folder, type):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(f"{folder}/{type}_confusion_matrix_{model_name}.png")
    plt.close()


def save_roc_curve(y_true, y_pred_prob, model_name, folder, type):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"{folder}/{type}_roc_curve_{model_name}.png")
    plt.close()


def save_all_plots(
    y_val_15, model_predictions, folder="latex/model_plots", type="evaluate"
):
    # Generate and save plots for each model
    for name in model_predictions:
        save_confusion_matrix(
            y_val_15, model_predictions[name]["y_pred"], name, folder=folder, type=type
        )
        save_roc_curve(
            y_val_15,
            model_predictions[name]["y_pred_prob"],
            name,
            folder=folder,
            type=type,
        )


def save_model_summary(
    model_summary: dict, file_path: str = "latex/model_summary.txt", type="evaluate"
):
    # Create the initial summary string with the type
    model_summary_str = f"Type: {type}\n"

    # Append each model and its metrics to the summary string
    for model, metrics in model_summary.items():
        model_summary_str += f"{model}: {metrics}\n"

    # Save the string to a text file
    with open(file_path, "w") as file:
        file.write(model_summary_str)


if __name__ == "__main__":
    # Create a directory for model plots
    os.makedirs("latex/model_plots", exist_ok=True)

    # TODO - Change this: Using a lighter version of the dataset for faster training
    df = load_dataset("code/creditcard_test_light09.csv")
    df = remove_duplicates(df)
    df = add_hour_columns(df)
    X, y = create_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_into_train_test(X, y)
    X_train_85, X_val_15, y_train_85, y_val_15 = train_validation_split(
        X_train, y_train
    )

    # Define a list of (model, name) tuples
    # Divide models into two groups: supervised and unsupervised and also by complexity (NN and ML)
    models_supervised_ML = [
        (RandomForestClassifier(n_estimators=2, random_state=42), "Random Forest"),
        (SVC(probability=True), "Support Vector Machine"),
    ]

    models_supervised_NN = [
        # (Sequential(), "Neural Network"),
    ]

    models_unsupervised_ML = [
        (RandomForestClassifier(n_estimators=2, random_state=42), "Random Forest"),
        (SVC(probability=True), "Support Vector Machine"),
    ]

    Models_unsupervised_NN = []  # TODO - Add neural network

    ### Evaluate the models

    # Define the hyperparameter space for each model
    params = {
        "Random Forest": {
            # "n_estimators": [2, 4, 6],
            "max_depth": [None, 4, 6, 8],
            "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [1, 2, 4],
        },
        "Support Vector Machine": {
            "C": [0.1],  # 1, 10, 100],  Commented for performance reasons
            # "gamma": ["scale", "auto"],
            # "kernel": ["linear", "rbf", "poly"],
        },
        "One Class SVM": {
            "gamma": ["scale"],  # , "auto"],
            # "kernel": ["linear", "rbf", "poly"],
        },
        "Isolation Forest": {
            "n_estimators": [2],  # , 200, 300],
            #  "max_samples": [100, 200, 300],
            "contamination": [0.1, 0.2, 0.3],
        },
    }

    # Evaluate the models using random search
    model_summary_evaluation, model_predictions_evaluation = evaluate_ml_models(
        models_supervised_ML,
        params,
        X_train_85,
        y_train_85,
        X_val_15,
        y_val_15,
        n_iter=2,
    )

    print(model_summary_evaluation)

    # Save plots
    save_all_plots(y_val_15, model_predictions_evaluation, type="evaluate")

    # Save model summary
    save_model_summary(
        model_summary_evaluation,
        file_path="latex/evaluate_model_summary.txt",
        type="evaluate",
    )

    ### Train the models

    # Train the models on the entire training set
    model_summary = {}
    model_predictions = {}

    model_summary, model_predictions = train_ml_models(
        X_train, y_train, X_test, y_test, model_summary_evaluation
    )

    print(model_summary)

    # Save plots
    save_all_plots(y_test, model_predictions, type="test")

    # Save model summary
    save_model_summary(
        model_summary, file_path="latex/test_model_summary.txt", type="test"
    )

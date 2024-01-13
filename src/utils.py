import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from joblib import dump
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.logger_cfg import app_logger


def load_config(filename):
    """Load YAML config file"""
    try:
        with open("cfg/cfg.yaml", "r") as file:
            app_logger.info("Config file loaded.")
            return yaml.safe_load(file)
    except FileNotFoundError:
        app_logger.error("Config file not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        app_logger.error(f"Error parsing YAML file: {exc}")
        sys.exit(1)
    except Exception as exc:
        app_logger.error(f"An unexpected error occurred: {exc}")
        sys.exit(1)


# Model evaluation summary
def save_model_summary(
    model_summary: dict, file_path: str = "latex/model_summary.txt", type="evaluate"
):
    """Save the model summary to a text file

    Args:
        model_summary (dict): Summary of the models
        file_path (str, optional): Path to the file. Defaults to "latex/model_summary.txt".
        type (str, optional): Type of the summary (evaluate or test). Defaults to "evaluate".
    """
    # Create the initial summary string with the type
    model_summary_str = f"Type: {type}\n"

    # Append each model and its metrics to the summary string
    for model, metrics in model_summary.items():
        model_summary_str += f"{model}: {metrics}\n"

    # Save the string to a text file
    try:
        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            file.write(model_summary_str)
            app_logger.info("Model summary saved.")
    except OSError as error:
        app_logger.info("Model summary can not be saved")
    else:
        app_logger.info("Successfully saved the model summary")


# Plots for the report
def save_confusion_matrix(y_true, y_pred, model_name, folder, type):
    """Save a confusion matrix plot

    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
        model_name (str): Name of the model
        folder (str): Folder to save the plot to
        type (str): Type of the plot (evaluate or test)
    """
    try:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate the percentage for each cell in the confusion matrix
        cm_perc = conf_matrix / conf_matrix.sum(keepdims=True) * 100

        # Define the display with display labels as the class names
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=["False", "True"]
        )

        # Plot using a specific color map and add title
        cm_display.plot(cmap=plt.cm.get_cmap("copper"))
        plt.title("Confusion Matrix")

        # Loop over data dimensions and create text annotations.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                percentage = f"{cm_perc[i, j]:.2f}%"  # Format as rounded integer
                plt.text(
                    j,
                    i,
                    f"\n\n({percentage})",
                    ha="center",
                    va="center",
                    color="white"
                    if conf_matrix[i, j] > conf_matrix.max() / 2
                    else "white",
                )
        plt.tight_layout()
        plt.savefig(f"{folder}/{type}_confusion_matrix_{model_name}.png")
        plt.close()

    except Exception as e:
        app_logger.info(
            f"An error occurred while saving the confusion matrix for {model_name}: {e}"
        )
        sys.exit(1)


def plot_precision_recall_curve(y_true, y_pred, folder, model_name, type):
    """
    Plots the Precision-Recall curve for given true labels and predicted probabilities.

    :param y_true: Array of true binary labels (0s and 1s)
    :param y_scores: Predicted probabilities for the positive class (between 0 and 1)
    """
    # Calculate precision-recall values
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # Calculate average precision score
    average_precision = average_precision_score(y_true, y_pred)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Average Precision = {average_precision:0.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"{folder}/{type}_precision_recal_curve_{model_name}.png")
    plt.close()


def save_roc_curve(y_true, y_pred_prob, model_name, folder, type):
    """Save a ROC curve plot

    Args:
        y_true (pd.Series): True labels
        y_pred_prob (pd.Series): Predicted probabilities
        model_name (str): Name of the model
        folder (str): Folder to save the plot to
        type (str): Type of the plot (evaluate or test)
    """
    try:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc_score = roc_auc_score(y_true, y_pred_prob)

        # Generate and save the ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.savefig(f"{folder}/{type}_roc_curve_{model_name}.png")
        plt.close()

    except Exception as e:
        app_logger.info(
            f"An error occurred while saving the ROC curve for {model_name}: {e}"
        )
        sys.exit(1)


def anomaly_scores(model_name, folder, scores="", type="evaluate"):
    try:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=50)
        plt.title("Anomaly Scores Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(f"{folder}/{type}_anomaly_scores_{model_name}.png")
        plt.close()
    except Exception as e:
        app_logger.info(
            f"An error occurred while saving the Anomaly Scores for {model_name}: {e}"
        )
        sys.exit(1)


def save_all_plots(
    y_val, model_predictions, folder="latex/model_plots", type="evaluate", scores=""
):
    """Save all plots for each model

    Args:
        y_val (pd.Series): Validation labels
        model_predictions (dict): Predictions for each model
        folder (str, optional): Folder to save the plots to. Defaults to "latex/model_plots".
        type (str, optional): Type of the plots (evaluate or test). Defaults to "evaluate".
    """

    # logger
    app_logger.info("Saving all plots")
    # Generate and save plots for each model
    for name in model_predictions:
        if (
            name != "OneClassSVM"
            and name != "IsolationForest"
            and name != "Autoencoder"
        ):
            save_roc_curve(
                y_val,
                model_predictions[name]["y_pred_prob"],
                name,
                folder=folder,
                type=type,
            )

        save_confusion_matrix(
            y_val,
            model_predictions[name]["y_pred"],
            name,
            folder=folder,
            type=type,
        )

        plot_precision_recall_curve(
            y_val,
            model_predictions[name]["y_pred"],
            folder=folder,
            model_name=name,
            type=type,
        )

        if name == "OneClassSVM" or name == "IsolationForest":
            anomaly_scores(name, folder, scores, type=type)


def save_model(model: object, model_name: str, directory: str = "bestmodels/"):
    """Save a model to a directory.

    Args:
        model (object): Model to save.
        model_name (str): Name of the model.
        directory (str, optional): Directory to save the model to. Defaults to "bestmodels/".
    """
    # Add model name to directory
    directory = f"{directory}{model_name}/"

    # Save the model
    try:
        os.makedirs(directory, exist_ok=True)
        if model_name == "NN":
            model.save(f"{directory}{model_name}.h5")
        else:
            dump(model, f"{directory}{model_name}.joblib")
    except OSError as error:
        print("Model '%s' can not be saved" % model_name)
    else:
        print("Successfully saved the model '%s'" % model_name)


def plot_all_metric_and_model_comparation():
    models_metrics = {}

    titles = [
        "ROC_AUC_Scores_Comparison",
        "F1_Scores_Comparison",
        "Accuracy_Comparison",
    ]

    with open("data_generated/test/test_all_summary.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Type"):  # Skip header or unrelated lines
                continue
            parts = line.split(": ", 1)
            if len(parts) == 2:
                model_name = parts[0].strip()
                metrics = eval(
                    parts[1].strip()
                )  # Convert string representation of dictionary to actual dictionary
                models_metrics[model_name] = metrics

    roc_auc_scores = {
        model: metrics["ROC AUC"] for model, metrics in models_metrics.items()
    }
    f1_scores = {
        model: metrics["F1 Score"] for model, metrics in models_metrics.items()
    }
    accuracies = {
        model: metrics["Accuracy"] for model, metrics in models_metrics.items()
    }
    colors = ["blue", "green", "red", "purple", "orange"]

    for metrics, title in zip([roc_auc_scores, f1_scores, accuracies], titles):
        names = list(metrics.keys())
        values = list(metrics.values())
        plt.figure(figsize=(10, 6))  # You can adjust the figure size
        plt.bar(names, values, color=colors[: len(metrics)])
        plt.ylabel("Score")
        plt.title(title.replace("_", " "))
        plt.xticks(rotation=45)
        plt.savefig(f"data_generated/test/plots/{title}.png")
        plt.close()


def save_outputs(model_predictions, name, folder="data_generated/test/outputs"):
    # if folder does not exist, create it

    if not os.path.exists(folder):
        os.makedirs(folder)

    csv_file = f"{folder}/_outputs_{name}.csv"
    # just save a array to csv file int format
    np.savetxt(csv_file, model_predictions, delimiter=",", fmt="%d")

    app_logger.info(f"Outputs saved to {csv_file}")

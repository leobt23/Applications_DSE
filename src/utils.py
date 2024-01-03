import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from joblib import dump
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

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
        cm = confusion_matrix(y_true, y_pred)

        # Plot and save the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        plt.savefig(f"{folder}/{type}_confusion_matrix_{model_name}.png")
        plt.close()

    except Exception as e:
        app_logger.info(
            f"An error occurred while saving the confusion matrix for {model_name}: {e}"
        )
        sys.exit(1)


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


def save_all_plots(
    y_val_15, model_predictions, folder="latex/model_plots", type="evaluate"
):
    """Save all plots for each model

    Args:
        y_val_15 (pd.Series): Validation labels
        model_predictions (dict): Predictions for each model
        folder (str, optional): Folder to save the plots to. Defaults to "latex/model_plots".
        type (str, optional): Type of the plots (evaluate or test). Defaults to "evaluate".
    """

    # logger
    app_logger.info("Saving all plots")
    # Generate and save plots for each model
    for name in model_predictions:
        if name == "One Class SVM" or name == "Isolation Forest":
            continue
        else:
            save_confusion_matrix(
                y_val_15,
                model_predictions[name]["y_pred"],
                name,
                folder=folder,
                type=type,
            )
            save_roc_curve(
                y_val_15,
                model_predictions[name]["y_pred_prob"],
                name,
                folder=folder,
                type=type,
            )


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

import sys

import yaml

from src.logger_cfg import app_logger


@staticmethod
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


@staticmethod
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
        with open(file_path, "w") as file:
            file.write(model_summary_str)
    except OSError as error:
        print("Model summary can not be saved")
    else:
        print("Successfully saved the model summary")

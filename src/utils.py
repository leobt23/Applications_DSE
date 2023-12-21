import sys

import yaml

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

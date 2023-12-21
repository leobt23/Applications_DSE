import sys

import yaml

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.logger_cfg import app_logger


def main():
    # Acess cfg file and get the path to the data file
    try:
        with open("cfg/cfg.yaml", "r") as file:
            cfg_file = yaml.safe_load(file)
            app_logger.info("Config file loaded.")
    except FileNotFoundError:
        app_logger.error("Config file not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        app_logger.error(f"Error parsing YAML file: {exc}")
        sys.exit(1)
    except Exception as exc:
        app_logger.error(f"An unexpected error occurred: {exc}")
        sys.exit(1)

    # Data Loading
    data_loader = DataLoader(cfg_file["data"]["light_version"])
    data = data_loader.load_csv()

    # Data Processing
    data_processor = DataProcessor(data)
    data_processor.remove_duplicates()
    data_processor.add_hour_columns()
    data_processor.split_into_features_and_targets()
    data_processor.split_into_train_test()
    X_train, X_test, X_val, y_train, y_test, y_val = data_processor.get_processed_data()


if __name__ == "__main__":
    main()

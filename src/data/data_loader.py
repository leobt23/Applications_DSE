import pandas as pd

from src.data.abstract_data_loader import AbstractDataLoader
from src.logger_cfg import app_logger


class DataLoader(AbstractDataLoader):
    def __init__(self, filepath: str):
        """Initialize the DataLoader with additional configurations.

        Args:
            filepath (str): The path to the data file.
            additional_param (optional): Additional parameter for customization.
        """
        super().__init__(filepath)  # Calls the __init__ method of the superclass

    def load_csv(self):
        """Load the csv file.

        Returns:
            pd.DataFrame: The data as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.filepath)
            app_logger.info(f"Data loaded from {self.filepath}")
            return df
        except FileNotFoundError:
            app_logger.error(f"File not found: {self.filepath}")
        except Exception as e:
            app_logger.error(f"An error occurred: {e}")

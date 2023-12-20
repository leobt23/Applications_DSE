import pandas as pd

from src.logger_cfg import app_logger


class DataLoader:
    """This class will load the data from a csv file. The path of the csv file will be given in a yaml file."""

    def __init__(self, filepath: str):
        """Initialize the DataLoader with the path to the data file.

        Args:
            filepath (str): The path to the data file.
        """

        self.filepath = filepath

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

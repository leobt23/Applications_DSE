import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger_cfg import app_logger


class DataProcessor:
    def __init__(self, data):
        self.data = data

    def remove_duplicates(self):
        """Remove duplicates from the data."""
        self.data = self.data.drop_duplicates()
        app_logger.info("Removing duplicates from the data.")

    def add_hour_columns(self):
        """
        Add hour columns to the dataframe based on 'Time'.
        """
        self.data["hour"] = self.data["Time"].apply(
            lambda x: np.ceil(float(x) / 3600) % 24
        )
        self.data["hour48"] = self.data["Time"].apply(
            lambda x: np.ceil(float(x) / 3600) % 48
        )
        app_logger.info("Adding hour columns to the data.")

    def split_into_features_and_targets(self):
        """
        Create feature (X) and label (y) sets from the dataframe.
        """
        # Sorting by 'Time'
        self.data = self.data.sort_values(by=["Time"])

        # Splitting the data into features and targets
        self.X = self.data.drop(["Class"], axis=1)
        self.y = self.data["Class"]
        app_logger.info("Splitting the data into features and targets.")

    def split_into_train_test(self):
        """
        Split the data into train, test and validation sets.
        """
        self.X_train = self.X[self.X["hour48"] < 24]
        self.y_train = self.y[self.X["hour48"] < 24]
        self.X_test = self.X[self.X["hour48"] >= 24]
        self.y_test = self.y[self.X["hour48"] >= 24]

        # Validation data is 15% of test data but keep the data in order by 'Time' of the test data.
        _, self.X_val, _, self.y_val = train_test_split(
            self.X_test, self.y_test, test_size=0.15, shuffle=False
        )

        self.X_train = self.X_train.drop(["hour48", "Time"], axis=1)
        self.X_test = self.X_test.drop(["hour48", "Time"], axis=1)

        app_logger.info("Splitting the data into train, test and validation sets.")

    def get_processed_data(self):  # -> pd.DataFrame:
        """
        Return the processed data.
        """
        return (
            self.X_train,
            self.X_test,
            self.X_val,
            self.y_train,
            self.y_test,
            self.y_val,
        )

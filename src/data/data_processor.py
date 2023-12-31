import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.abstract_data_processor import AbstractDataProcessor
from src.logger_cfg import app_logger


class DataProcessor(AbstractDataProcessor):
    def __init__(self, data):
        """Initialize the DataProcessor with additional configurations.

        Args:
            data (pd.DataFrame): The data as a pandas DataFrame.
        """
        super().__init__(data)

    def remove_duplicates(self):
        """Remove duplicates from the data."""
        self.data = self.data.drop_duplicates()
        app_logger.info("Removing duplicates from the data.")

    def add_hour_columns(self):
        """
        Add hour columns to the dataframe based on 'Time'.
        """
        self.data = self.data.copy()
        self.data["hour"] = self.data["Time"].apply(
            lambda x: np.ceil(float(x) / 3600) % 24
        )
        self.data["hour48"] = self.data["Time"].apply(
            lambda x: np.ceil(float(x) / 3600) % 48
        )
        app_logger.info("Adding hour columns to the data.")

    def scaling(self):
        """
        Scale the data.
        """
        app_logger.info("Scaling the data.")

        # Logarithm of 10 to the Amount column
        self.data["Amount"] = np.log10(self.data["Amount"] + 1)

        # Separate out the features for scaling
        features = self.data.iloc[:, :-3]  # All columns except last three
        scaler = StandardScaler()

        # Fit and transform the features
        scaled_features = scaler.fit_transform(features)

        # Update the data with scaled features
        self.data.update(pd.DataFrame(scaled_features, columns=features.columns))

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
        self.X_val = self.X_val.drop(["hour48", "Time"], axis=1)

        app_logger.info("Splitting the data into train, test and validation sets.")

    def get_processed_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
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

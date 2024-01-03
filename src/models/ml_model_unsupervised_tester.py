from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger, save_all_plots


class MLModelTester(AbstractModelTester):
    """Test ML models"""

    def __init__(self):
        super().__init__()

    def get_model(self, model_name: str) -> Tuple:
        """Load model from joblib file

        Args:
            model_name (str): model name

        Returns:
            Tuple: model
        """
        # load joblib file as model
        # TODO: change path to config file
        path_to_model = f"data_generated/evaluation/{model_name}/{model_name}.joblib"
        model = joblib.load(path_to_model)

        return model

    def test_ml_models_supervised(
        self,
        models: dict,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        model_summary_evaluation: dict,
    ) -> Tuple:
        """Test ML models

        Args:
            models (dict): models to be tested
            X_test (pd.DataFrame): test features
            y_test (pd.DataFrame): test targets
            model_summary_evaluation (dict): model summary evaluation

        Returns:
            Tuple: model summary, model predictions
        """
        model_summary = {}
        model_predictions = {}

        for _, name in models:
            if name in model_summary_evaluation:
                app_logger.info(f"Testing {name}...")

            # Get model
            model = self.get_model(name)

            # Make predictions
            y_pred = model.predict(X_test)
            # Convert anomaly labels (-1, 1) to binary labels (0, 1)
            y_pred_binary = np.where(
                y_pred == -1, 1, 0
            )  # Assuming 1 for anomalies, 0 for normal

            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            # Add model performance to summary
            model_summary[name] = {"F1 Score": f1, "Accuracy": accuracy}

            model_predictions[name] = y_pred_binary

            save_all_plots(
                y_test,
                model_predictions,
                folder="data_generated/test/plots",
                type="test",
            )

        return model_summary, model_predictions

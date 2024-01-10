from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger, save_all_plots


class MLModelUnsupervisedTester(AbstractModelTester):
    """Test ML models"""

    def __init__(
        self, models, X_train, X_test, y_test, model_summary, model_predictions
    ):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_summary = model_summary
        self.model_predictions = model_predictions
        self.models = models

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

    def predict_model(self, model, name, model_summary, model_predictions):
        # Make predictions (anomaly detection)
        y_pred = model.predict(self.X_test)

        # Convert anomaly labels (-1, 1) to binary labels (0, 1)
        y_pred_binary = np.where(
            y_pred == -1, 1, 0
        )  # Assuming 1 for anomalies, 0 for normal

        # Calculate metrics (adapted for anomaly detection)
        f1 = f1_score(self.y_test, y_pred_binary)
        accuracy = accuracy_score(self.y_test, y_pred_binary)
        roc_auc = roc_auc_score(self.y_test, y_pred_binary)

        # assuming the model has a decision_function or score_samples method
        if hasattr(model, "decision_function"):
            scores = model.decision_function(self.X_test)
        else:
            scores = model.score_samples(self.X_test)

        # Add model performance to summary
        model_summary[name] = {"ROC AUC": roc_auc, "F1 Score": f1, "Accuracy": accuracy}
        model_summary[name]["Best Params"] = model.get_params()
        model_predictions[name] = {
            "y_pred": y_pred_binary,
        }

        return model_summary, model_predictions, scores

    def test_model(self):
        model_summary = {}
        model_predictions = {}
        for _, name in self.models:
            print(f"Testing {name}...")

            model = self.get_model(name)

            # Make predictions
            model_summary, model_predictions, scores = self.predict_model(
                model, name, model_summary, model_predictions
            )

            save_all_plots(
                self.y_test,
                model_predictions,
                folder="data_generated/test/plots",
                type="test",
                scores=scores,
            )

        return model_summary, model_predictions

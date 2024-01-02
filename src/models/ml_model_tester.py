from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger


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
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_prob)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            # Add model performance to summary
            model_summary[name] = {
                "ROC AUC": auc_score,
                "F1 Score": f1,
                "Accuracy": accuracy,
                "Best Params": model_summary_evaluation[name]["Best Params"],
            }

            model_predictions[name] = {"y_pred": y_pred, "y_pred_prob": y_pred_prob}

        return model_summary, model_predictions

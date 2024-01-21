from typing import Tuple

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger, save_all_plots, save_outputs


class NNModelTester(AbstractModelTester):
    def __init__(self):
        super().__init__()

    def get_model(self, model_path: str = "data_generated/evaluation/NN/NN.h5"):
        """Load model from joblib file

        Args:
            model_path (str): Path to the model file.

        Returns:
            object: Loaded model object.
        """
        return load_model(model_path)

    def test_nn_model(
        self,
        X_test,
        y_test,
    ) -> Tuple:
        """Test NN model

        Args:
            X_test (pd.DataFrame): test features
            y_test (pd.DataFrame): test targets

        Returns:
            Tuple: model summary, model predictions
        """
        model_summary = {}
        model_predictions = {}

        app_logger.info(f"Testing NN...")

        # Get model
        model_nn = self.get_model(model_path="data_generated/evaluation/NN/NN.h5")

        y_pred_nn = (model_nn.predict(X_test) > 0.5).astype(int).ravel()
        y_pred_prob = model_nn.predict(X_test)
        save_outputs(
            y_pred_prob, "NN_probs" + "0", folder="data_generated/test/outputs"
        )
        y_pred_prob = y_pred_prob.ravel()
        auc_nn = roc_auc_score(y_test, y_pred_prob)
        f1_nn = f1_score(y_test, y_pred_nn)
        accuracy_nn = accuracy_score(y_test, y_pred_nn)

        # Convert probabilities to binary predictions using a threshold
        precision = average_precision_score(y_test, y_pred_prob)
        # Calculate recall
        recall = recall_score(y_test, y_pred_nn)

        # Add neural network performance to summary
        model_summary["Neural Network"] = {
            "ROC AUC": auc_nn,
            "F1 Score": f1_nn,
            "Accuracy": accuracy_nn,
            "Precison": precision,
            "Recall": recall,
        }

        model_predictions["Neural Network"] = {
            "y_pred": y_pred_nn,
            "y_pred_prob": y_pred_prob,
        }

        save_all_plots(
            y_test,
            model_predictions,
            folder="data_generated/test/plots",
            type="test",
        )

        save_outputs(y_pred_nn, "NN", folder="data_generated/test/outputs")

        return model_summary, model_predictions

from typing import Tuple

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger, save_all_plots


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
        y_pred_prob = y_pred_prob.ravel()
        auc_nn = roc_auc_score(y_pred_nn, y_pred_nn)
        f1_nn = f1_score(y_pred_nn, y_pred_nn)
        accuracy_nn = accuracy_score(y_pred_nn, y_pred_nn)

        # Add neural network performance to summary
        model_summary["Neural Network"] = {
            "ROC AUC": auc_nn,
            "F1 Score": f1_nn,
            "Accuracy": accuracy_nn,
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

        return model_summary, model_predictions

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils import save_model


class MLModelUnsupervisedEvaluator:  # TODO: AbstractModelEvaluator
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

        # Add model performance to summary
        model_summary[name] = {"F1 Score": f1, "Accuracy": accuracy}
        model_predictions[name] = y_pred_binary

        return model_summary, model_predictions

    def fit_model(self):
        model_summary = {}
        model_predictions = {}
        for model, name in self.models:
            print(f"Training {name}...")

            # Train the model on normal data
            model.fit(self.X_train)

            # Make predictions
            model_summary, model_predictions = self.predict_model(
                model, name, model_summary, model_predictions
            )

            # Save the model
            path_save_model = f"data_generated/evaluation/"
            save_model(model=model, model_name=name, directory=path_save_model)

        return model_summary, model_predictions

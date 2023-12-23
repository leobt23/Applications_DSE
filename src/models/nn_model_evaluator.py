# nn_model_evaluator.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.logger_cfg import app_logger


class NNModelEvaluator:
    """
    Class to evaluate NN models.
    """

    def __init__(self):
        """
        Args:
            models (list): List of tuples containing the model and its name.
            params (dict): Dictionary containing the parameters to be used in the RandomizedSearchCV.
        """
        super().__init__()
        self.model_nn = None
        self.history = None
        self.model_summary = None
        self.model_predictions = None

    def build_model(self, input_dim):
        """Builds a NN model.

        Args:
            input_dim (int): Input dimension.

        Returns:
            tensorflow.keras.Sequential: NN model.
        """
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def compile_and_train_model(self, model, X_train, y_train):
        """Compiles and trains a NN model.

        Args:
            model (tensorflow.keras.Sequential): NN model.
            X_train (np.array): Training features.
            y_train (np.array): Training targets.

        Returns:
            tensorflow.keras.callbacks.History: Training history.
        """
        model.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
        )
        callback = EarlyStopping(monitor="val_loss", patience=5)
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=128,
            callbacks=[callback],
        )
        return history

    def evaluate_performance(self, model, X_val, y_val):
        """Evaluates a given model.

        Args:
            model (tensorflow.keras.Sequential): NN model.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.

        Returns:
            dict: Dictionary containing the evaluation metrics.
        """
        y_pred = (model.predict(X_val) > 0.5).astype(int).ravel()
        y_pred_prob = model.predict(X_val).ravel()
        return {
            "ROC AUC": roc_auc_score(y_val, y_pred),
            "F1 Score": f1_score(y_val, y_pred),
            "Accuracy": accuracy_score(y_val, y_pred),
        }, {"y_pred": y_pred, "y_pred_prob": y_pred_prob}

    def evaluate(self, X_train, y_train, X_val, y_val):
        """Evaluates the model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training targets.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.
        """
        model = self.build_model(X_train.shape[1])
        history = self.compile_and_train_model(model, X_train, y_train)
        model_summary, model_predictions = self.evaluate_performance(
            model, X_val, y_val
        )
        self.model_nn, self.history, self.model_summary, self.model_predictions = (
            model,
            history,
            model_summary,
            model_predictions,
        )

    def get_evaluation_results(self):
        """Returns the evaluation results.

        Returns:
            tensorflow.keras.Sequential: NN model.
            tensorflow.keras.callbacks.History: Training history.
            dict: Dictionary containing the evaluation metrics.
            dict: Dictionary containing the model predictions.
        """
        return self.model_nn, self.history, self.model_summary, self.model_predictions

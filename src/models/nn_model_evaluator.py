# nn_model_evaluator.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.logger_cfg import app_logger
from src.models.abstract_model_evaluator import AbstractModelEvaluator
from src.utils import save_all_plots, save_model


class NNModelEvaluator(AbstractModelEvaluator):
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

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))

        # Add title
        plt.suptitle("Training History", fontsize=16)
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")

        plt.savefig("data_generated/evaluation/plots/nn.png")  # Add your path here
        plt.close()

    def build_model(self, input_dim: int):
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
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        callback = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=128,
            callbacks=[callback],
        )
        return history, model

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
            "NN": {
                "ROC AUC": roc_auc_score(y_val, y_pred_prob),
                "F1 Score": f1_score(y_val, y_pred),
                "Accuracy": accuracy_score(y_val, y_pred),
            }
        }, {"NN": {"y_pred": y_pred, "y_pred_prob": y_pred_prob}}

    def evaluate(self, X_train, y_train, X_val, y_val):
        """Evaluates the model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training targets.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.
        """
        model = self.build_model(X_train.shape[1])
        history, model = self.compile_and_train_model(model, X_train, y_train)
        model_summary, model_predictions = self.evaluate_performance(
            model, X_val, y_val
        )

        # Save model
        path_save_model = f"data_generated/evaluation/"
        save_model(model, "NN", directory=path_save_model)

        self.model_nn, self.history, self.model_summary, self.model_predictions = (
            model,
            history,
            model_summary,
            model_predictions,
        )

        save_all_plots(
            y_val,
            model_predictions,
            folder="data_generated/evaluation/plots",
            type="validation",
        )

        self.plot_training_history(history)

    def get_evaluation_results(self) -> tuple:
        """Returns the evaluation results.

        Returns:
            tensorflow.keras.Sequential: NN model.
            tensorflow.keras.callbacks.History: Training history.
            dict: Dictionary containing the evaluation metrics.
            dict: Dictionary containing the model predictions.
        """
        return self.model_nn, self.history, self.model_summary, self.model_predictions

# nn_model_evaluator.py
import kerastuner as kt
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
        self.input_shape = 0

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

    def train_model(self, model, X_train, y_train):
        """Compiles and trains a NN model.

        Args:
            model (tensorflow.keras.Sequential): NN model.
            X_train (np.array): Training features.
            y_train (np.array): Training targets.

        Returns:
            tensorflow.keras.callbacks.History: Training history.
        """

        callback = EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=500,
            batch_size=64,
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

    def build_model(self, hp):
        model = Sequential()
        # First Dense layer
        model.add(
            Dense(
                units=hp.Int("units1", min_value=32, max_value=256, step=32),
                activation="relu",
                input_shape=(self.input_shape,),
            )
        )  # input shape inferred
        # Dropout layer after the first Dense layer
        model.add(
            Dropout(rate=hp.Float("dropout1", min_value=0.0, max_value=0.5, step=0.25))
        )

        # Second Dense layer
        model.add(
            Dense(
                units=hp.Int("units2", min_value=32, max_value=256, step=32),
                activation="relu",
            )
        )

        model.add(Dense(1, activation="sigmoid"))  # Example output layer

        model.compile(
            optimizer=hp.Choice("optimizer", values=["adam", "sgd"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def evaluate(self, X_train, y_train, X_val, y_val):
        """Evaluates the model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training targets.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.
        """

        self.input_shape = X_train.shape[1]

        tuner = kt.Hyperband(
            self.build_model,
            objective="val_accuracy",  # Assuming you have a validation set to evaluate on
            max_epochs=10,  # Maximum number of epochs to train one model. Adjust as necessary.
            factor=3,  # Reduction factor for the number of epochs and number of models for each bracket.
        )

        # Execute the search
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

        # Get the best hyperparameters and rebuild the model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.build_model(best_hps)

        history, model = self.train_model(model, X_train, y_train)
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

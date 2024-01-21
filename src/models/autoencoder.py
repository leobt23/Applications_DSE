from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# f1 score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from src.utils import app_logger, save_confusion_matrix, save_model, save_outputs


class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        app_logger.info("Creating Autoencoder model.")

    def plot_training_history(self, history):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training vs. Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig("data_generated/evaluation/plots/autoencoder.png")
        plt.close()

    def create_callback(self):
        # define our early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        save_model = tf.keras.callbacks.ModelCheckpoint(
            filepath="data_generated/evaluation/Autoencoder/autoencoder_best_weights.hdf5",
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
            mode="min",
        )

        # callbacks argument only takes a list
        cb = [early_stop, save_model]

        return cb

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(
            Dense(self.input_dim, activation="elu", input_shape=(self.input_dim,))
        )
        encoder.add(Dense(16, activation="elu"))
        encoder.add(Dense(8, activation="elu"))
        encoder.add(Dense(4, activation="elu"))
        encoder.add(Dense(self.encoding_dim, activation="elu"))
        return encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(Dense(4, activation="elu", input_shape=(self.encoding_dim,)))
        decoder.add(Dense(8, activation="elu"))
        decoder.add(Dense(16, activation="elu"))
        decoder.add(Dense(self.input_dim, activation="elu"))
        return decoder

    def compile(self):
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(
            optimizer="adam", loss="mse", metrics=["acc"]
        )  # TODO: loss="binary_crossentropy" check this

    def train(self, X_train, X_val, epochs, batch_size):
        cb = self.create_callback()

        history = self.autoencoder.fit(
            X_train,
            X_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            validation_data=(X_val, X_val),
        )

        self.plot_training_history(history)

    def calculate_metrics(
        self, X_test, reconstructions, y_test, model_summary, name, model_predictions
    ):
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

        mse.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        clean = mse[y_test == 0]
        fraud = mse[y_test == 1]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.hist(clean, bins=200, density=True, label="clean", alpha=0.6, color="green")
        ax.hist(fraud, bins=200, density=True, label="fraud", alpha=0.6, color="red")

        plt.title("Distribution of the Reconstruction Loss")
        plt.legend()
        plt.savefig("data_generated/test/plots/autoencoder_reconstruction_loss.png")
        plt.close()

        THRESHOLD = 6

        def mad_score(points):
            m = np.median(points)
            ad = np.abs(points - m)
            mad = np.median(ad)

            return 0.6745 * ad / mad

        z_scores = mad_score(mse)
        outliers = z_scores > THRESHOLD

        save_confusion_matrix(
            y_test,
            outliers,
            model_name="Autoencoder",
            folder="data_generated/test/plots",
            type="test",
        )

        # get (mis)classification
        cm = confusion_matrix(y_test, outliers)

        # true/false positives/negatives
        (tn, fp, fn, tp) = cm.flatten()

        extra_notes = f"Detected {np.sum(outliers):,} outliers in a total of {np.size(z_scores):,} transactions [{np.sum(outliers)/np.size(z_scores):.2%}]."

        extra_notes += f"""MAD method with threshold={THRESHOLD} are as follows: {cm}
            % of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
            % of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}"""

        # Calculate metrics (adapted for anomaly detection)
        f1 = f1_score(y_test, outliers)
        accuracy = accuracy_score(y_test, outliers)
        roc_auc = roc_auc_score(y_test, outliers)

        # Add model performance to summary
        model_summary[name] = {"ROC AUC": roc_auc, "F1 Score": f1, "Accuracy": accuracy}
        # Best Params not available for Autoencoder, NA as string
        model_summary[name]["Best Params"] = "NA"
        model_summary[name]["Extra Notes"] = extra_notes
        model_predictions[name] = {
            "y_pred": outliers,
        }

        save_outputs(outliers, "Autoencoder", folder="data_generated/test/outputs")

        return model_summary, model_predictions

    def predict(self, x, y):
        model_summary = {}
        model_predictions = {}
        pred_output = self.autoencoder.predict(x)
        model_summary, model_predictions = self.calculate_metrics(
            x, pred_output, y, model_summary, "Autoencoder", model_predictions
        )
        save_model(
            self.autoencoder, "Autoencoder", "data_generated/evaluation/Autoencoder/"
        )
        return model_summary, model_predictions

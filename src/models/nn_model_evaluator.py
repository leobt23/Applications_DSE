# nn_model_evaluator.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class NNModelEvaluator:
    def evaluate(
        self, X_train, y_train, X_val, y_val, model_summary, model_predictions, model_nn
    ):
        model_nn.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
        model_nn.add(Dropout(0.5))
        model_nn.add(Dense(64, activation="relu"))
        model_nn.add(Dense(1, activation="sigmoid"))
        model_nn.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
        )

        callback = EarlyStopping(monitor="val_loss", patience=5)
        history = model_nn.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=128,
            callbacks=[callback],
        )

        y_pred = (model_nn.predict(X_val) > 0.5).astype(int).ravel()
        y_pred_prob = model_nn.predict(X_val).ravel()

        model_summary["Neural Network"] = {
            "ROC AUC": roc_auc_score(y_val, y_pred),
            "F1 Score": f1_score(y_val, y_pred),
            "Accuracy": accuracy_score(y_val, y_pred),
        }
        model_predictions["Neural Network"] = {
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
        }

        return model_nn, history, model_summary, model_predictions

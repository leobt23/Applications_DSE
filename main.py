import random as rn
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# classification report
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC, OneClassSVM
from tensorflow.keras.models import Sequential

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.resampling import Resampler
from src.models.autoencoder import Autoencoder
from src.models.ml_model_evaluator import MLModelEvaluator
from src.models.ml_model_tester import MLModelTester
from src.models.ml_model_unsupervised_evaluator import MLModelUnsupervisedEvaluator
from src.models.ml_model_unsupervised_tester import MLModelUnsupervisedTester
from src.models.nn_model_evaluator import NNModelEvaluator
from src.models.nn_model_tester import NNModelTester
from src.utils import (
    load_config,
    plot_all_metric_and_model_comparation,
    save_model_summary,
)


def main():
    # setting random seeds for libraries to ensure reproducibility
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)

    # Acess cfg file and get the path to the data file
    cfg_file = load_config("config.yml")

    # Data Loading
    data_loader = DataLoader(cfg_file["data"]["light_version"])
    data = data_loader.load_csv()

    # Data Processing
    data_processor = DataProcessor(data)
    data_processor.remove_duplicates()
    data_processor.add_hour_columns()
    data_processor.log_amount()
    data_processor.split_into_features_and_targets()
    data_processor.split_into_train_test()
    data_processor.scale_data()
    X_train, X_test, X_val, y_train, y_test, y_val = data_processor.get_processed_data()

    resampler = Resampler()

    X_train, y_train = resampler.apply_resampling(
        cfg_file["resampling"]["strategy"],
        X_train,
        y_train,
        sampling_strategy=cfg_file["resampling"].get("sampling_strategy", "auto"),
    )
    """
    # Define models
    models_supervised_ML = [
        (
            RandomForestClassifier(n_estimators=2, random_state=42),
            "RandomForestClassifier",
        ),
        (SVC(probability=True), "SVC"),
    ]

    models_unsupervised_ML = [
        (OneClassSVM(), "OneClassSVM"),
        (IsolationForest(), "IsolationForest"),
    ]

    models_supervised_NN = Sequential()

    # ML Model Evaluation
    ml_evaluator = MLModelEvaluator(
        models_supervised_ML, cfg_file["models_parameters"]["ml_supervised"]
    )

    ml_evaluator.evaluate(X_train, y_train, X_val, y_val, n_iter=2)
    
    # Retrieve summaries and predictions
    model_summary, model_predictions = ml_evaluator.get_evaluation_results()

    ml_evaluator_unsupervised = MLModelUnsupervisedEvaluator(
        models_unsupervised_ML,
        X_train,
        y_train,
        X_val,
        y_val,
        model_summary,
        model_predictions,
        cfg_file["models_parameters"]["ml_unsupervised"],
    )
    _model_summary, _model_predictions = ml_evaluator_unsupervised.fit_model()
    model_summary.update(_model_summary)

    # Initialize NN Model Evaluator
    nn_evaluator = NNModelEvaluator()

    # Evaluate NN Model
    nn_evaluator.evaluate(X_train, y_train, X_val, y_val)

    # Retrieve summaries and predictions
    (
        models_supervised_NN,
        history,
        _model_summary,
        model_predictions,
    ) = nn_evaluator.get_evaluation_results()

    model_summary.update(_model_summary)

    save_model_summary(model_summary, file_path=cfg_file["model_summary_validation"])

    # Test ML Models
    ml_tester = MLModelTester()
    model_summary, model_predictions = ml_tester.test_ml_models_supervised(
        models_supervised_ML, X_test, y_test, model_summary
    )

    ml_unsupervised_tester = MLModelUnsupervisedTester(
        models_unsupervised_ML,
        X_train,
        X_test,
        y_test,
        model_summary,
        model_predictions,
    )

    _model_summary, model_predictions = ml_unsupervised_tester.test_model()
    model_summary.update(_model_summary)

    # Test NN Model
    nn_tester = NNModelTester()
    _model_summary, model_predictions = nn_tester.test_nn_model(X_test, y_test)

    model_summary.update(_model_summary)
    save_model_summary(model_summary, file_path=cfg_file["model_summary_test"])

    plot_all_metric_and_model_comparation()
    """

    # Train an autoencoder on the training data
    autoencoder = Autoencoder(input_dim=X_train.shape[1], encoding_dim=2)
    autoencoder.compile()
    autoencoder.train(X_train, X_val, epochs=50, batch_size=256)

    reconstructions = autoencoder.predict(X_test)

    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    clean = mse[y_test == 0]
    fraud = mse[y_test == 1]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.hist(clean, bins=50, density=True, label="clean", alpha=0.6, color="green")
    ax.hist(fraud, bins=50, density=True, label="fraud", alpha=0.6, color="red")

    plt.title("(Normalized) Distribution of the Reconstruction Loss")
    plt.legend()
    # save plot
    plt.savefig("data_generated/test/plots/autoencoder_reconstruction_loss.png")
    plt.close()

    THRESHOLD = 3

    def mad_score(points):
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad

    z_scores = mad_score(mse)
    outliers = z_scores > THRESHOLD

    print(
        f"Detected {np.sum(outliers):,} outliers in a total of {np.size(z_scores):,} transactions [{np.sum(outliers)/np.size(z_scores):.2%}]."
    )

    from sklearn.metrics import confusion_matrix, precision_recall_curve

    # get (mis)classification
    cm = confusion_matrix(y_test, outliers)

    # true/false positives/negatives
    (tn, fp, fn, tp) = cm.flatten()

    print(
        f"""The classifications using the MAD method with threshold={THRESHOLD} are as follows:
{cm}

% of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
% of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}"""
    )

    """
    # get f1 score, auc score, precision, recall, accuracy
    _f1_score = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    autoencoder_classification = {
        "Autoencoder": {
            "f1_score": _f1_score,
            "auc_score": auc_score,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }
    }
    print(autoencoder_classification)
    # model_summary.update(autoencoder_classification)
    """


if __name__ == "__main__":
    main()

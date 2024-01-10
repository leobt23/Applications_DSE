import random as rn
import sys

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
    data_loader = DataLoader(cfg_file["data"]["full_version"])
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

    autoencoder = Autoencoder(input_dim=X_train.shape[1], encoding_dim=2)
    autoencoder.compile()
    autoencoder.train(X_train, X_val, epochs=100, batch_size=256)
    X_train_encoded = autoencoder.encode(X_train)
    X_test_encoded = autoencoder.encode(X_test)
    # decoded_data = autoencoder.decode(encoded_data)

    # save data
    np.savetxt(
        "data/processed_data/X_train_encoded_data.csv", X_train_encoded, delimiter=","
    )
    np.savetxt("data/processed_data/X_test_encoded.csv", X_test_encoded, delimiter=",")

    
    # Train a classifier on the encoded data
    classifier = RandomForestClassifier(
        min_samples_split=10, max_features="log2", max_depth=8
    )
    classifier.fit(X_train_encoded, y_train)

    # Make predictions and evaluate the classifier
    y_pred = classifier.predict(X_test_encoded)

    
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

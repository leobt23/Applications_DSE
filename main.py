import sys

import yaml
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from tensorflow.keras.models import Sequential

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.resampling import Resampler
from src.logger_cfg import app_logger
from src.models.ml_model_evaluator import MLModelEvaluator
from src.models.nn_model_evaluator import NNModelEvaluator
from src.utils import load_config


def main():
    # Acess cfg file and get the path to the data file
    cfg_file = load_config("config.yml")

    # Data Loading
    data_loader = DataLoader(cfg_file["data"]["light_version"])
    data = data_loader.load_csv()

    # Data Processing
    data_processor = DataProcessor(data)
    data_processor.remove_duplicates()
    data_processor.add_hour_columns()
    data_processor.split_into_features_and_targets()
    data_processor.split_into_train_test()
    X_train, X_test, X_val, y_train, y_test, y_val = data_processor.get_processed_data()

    # Apply resampling - TODO: SMOTE is not working IDK why;
    resampler = Resampler()
    X_train, y_train = resampler.apply_resampling(
        cfg_file["resampling"]["strategy"],
        X_train,
        y_train,
        sampling_strategy=cfg_file["resampling"].get("sampling_strategy", "auto"),
    )

    # Define models
    models_supervised_ML = [
        (RandomForestClassifier(n_estimators=2, random_state=42), "Random Forest"),
        (SVC(probability=True), "Support Vector Machine"),
    ]

    # ML Model Evaluation
    ml_evaluator = MLModelEvaluator(
        models_supervised_ML, cfg_file["models_parameters"]["ml_supervised"]
    )
    models_supervised_NN = Sequential()

    ml_evaluator.evaluate(X_train, y_train, X_val, y_val, n_iter=2)

    # Retrieve summaries and predictions
    model_summary, model_predictions = ml_evaluator.get_evaluation_results()

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

    print(model_summary)


if __name__ == "__main__":
    main()

import sys

import yaml
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from tensorflow.keras.models import Sequential

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.resampling import Resampler
from src.models.ml_model_evaluator import MLModelEvaluator
from src.models.ml_model_tester import MLModelTester
from src.models.ml_model_unsupervised_evaluator import MLModelUnsupervisedEvaluator
from src.models.ml_model_unsupervised_tester import MLModelUnsupervisedTester
from src.models.nn_model_evaluator import NNModelEvaluator
from src.models.nn_model_tester import NNModelTester
from src.utils import load_config, save_model_summary


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
    data_processor.log_amount()
    data_processor.split_into_features_and_targets()
    data_processor.split_into_train_test()
    data_processor.scale_data()
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


if __name__ == "__main__":
    main()

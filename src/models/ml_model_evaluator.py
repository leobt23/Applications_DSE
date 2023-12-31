from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.logger_cfg import app_logger
from src.models.abstract_model_evaluator import AbstractModelEvaluator
from src.utils import save_all_plots, save_model, save_model_summary


class MLModelEvaluator(AbstractModelEvaluator):
    """
    Class to evaluate ML models.
    """

    def __init__(self, models, params):
        """
        Args:
            models (list): List of tuples containing the model and its name.
            params (dict): Dictionary containing the parameters to be used in the RandomizedSearchCV.
        """
        super().__init__()
        self.models = models
        self.params = params
        self.model_summary = None
        self.model_predictions = None

    def perform_random_search(self, model, X_train, y_train, n_iter, cv):
        """Performs a RandomizedSearchCV for a given model.

        Args:
            model (sklearn model): Model to be evaluated.
            X_train (np.array): Training features.
            y_train (np.array): Training targets.
            n_iter (int): Number of iterations to be performed.
            cv (int): Number of cross-validation folds.

        Returns:
            sklearn model: Best model found by the RandomizedSearchCV.
        """
        random_search = RandomizedSearchCV(
            model,
            self.params[model.__class__.__name__],
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )

        random_search.fit(X_train, y_train)

        model_name = model.__class__.__name__
        # Add the model name to path_to_save
        path_to_save = f"data_generated/evaluation/{model_name}/randsearch_summary.txt"

        save_model_summary(
            random_search.cv_results_, file_path=path_to_save, type="Random Search"
        )
        random_search.fit(X_train, y_train)
        return random_search

    def evaluate_model(self, model, X_val, y_val):
        """Evaluates a given model.

        Args:
            model (sklearn model): Model to be evaluated.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.

        Returns:
            dict: Dictionary containing the evaluation metrics.
        """
        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        return {
            "ROC AUC": roc_auc_score(y_val, y_pred_prob),
            "F1 Score": f1_score(y_val, y_pred),
            "Accuracy": accuracy_score(y_val, y_pred),
        }

    def evaluate(self, X_train, y_train, X_val, y_val, n_iter=100, cv=5):
        """Evaluates all models.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training targets.
            X_val (np.array): Validation features.
            y_val (np.array): Validation targets.
            n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
        """
        model_summary = {}
        model_predictions = {}

        for model, name in self.models:
            app_logger.info(f"Evaluating {name}...")
            random_search = self.perform_random_search(
                model, X_train, y_train, n_iter, cv
            )
            best_model = random_search.best_estimator_
            model_eval = self.evaluate_model(best_model, X_val, y_val)

            model_summary[name] = model_eval
            model_summary[name]["Best Params"] = random_search.best_params_
            model_predictions[name] = {
                "y_pred": best_model.predict(X_val),
                "y_pred_prob": best_model.predict_proba(X_val)[:, 1],
            }

            # Save model
            path_save_model = f"data_generated/evaluation/"
            save_model(model, name, directory=path_save_model)

        self.model_summary = model_summary
        self.model_predictions = model_predictions

        save_all_plots(
            y_val,
            model_predictions,
            folder="data_generated/evaluation/plots",
            type="validation",
        )

    def get_evaluation_results(self):
        """Returns the evaluation results.

        Returns:
            dict: Dictionary containing the evaluation results.
        """
        return self.model_summary, self.model_predictions

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


class MLModelEvaluator:
    def __init__(self, models, params):
        self.models = models
        self.params = params

    def evaluate(self, X_train, y_train, X_val, y_val, n_iter=100, cv=5):
        model_summary = {}
        model_predictions = {}

        for model, name in self.models:
            print(f"Evaluating {name}...")
            random_search = RandomizedSearchCV(
                model,
                self.params[name],
                n_iter=n_iter,
                cv=cv,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1,
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_

            y_pred = best_model.predict(X_val)
            y_pred_prob = best_model.predict_proba(X_val)[:, 1]

            model_summary[name] = {
                "ROC AUC": roc_auc_score(y_val, y_pred_prob),
                "F1 Score": f1_score(y_val, y_pred),
                "Accuracy": accuracy_score(y_val, y_pred),
                "Best Params": random_search.best_params_,
            }
            model_predictions[name] = {"y_pred": y_pred, "y_pred_prob": y_pred_prob}

        return model_summary, model_predictions

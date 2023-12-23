from abc import ABC, abstractmethod


class AbstractModelEvaluator(ABC):
    """
    Abstract class to evaluate models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def get_evaluation_results(self):
        pass

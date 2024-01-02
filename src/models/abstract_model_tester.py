from abc import ABC, abstractmethod


class AbstractModelTester(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_model():
        pass

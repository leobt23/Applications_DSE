from abc import ABC, abstractmethod


class AbstractDataProcessor(ABC):
    def __init__(self, data):
        """
        Abstract class for data processing.
        """
        self.data = data

    @abstractmethod
    def get_processed_data(self):
        """
        Abstract method for getting the processed data.
        """
        pass

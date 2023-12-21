from abc import ABC, abstractmethod


class AbstractDataLoader(ABC):
    def __init__(self, filepath):
        """Initialize the DataLoader with the path to the data file.

        Args:
            data (str): The path to the data file.
        """
        self.filepath = filepath

    @abstractmethod
    def load_csv(self):
        pass

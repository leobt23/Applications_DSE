from abc import ABC, abstractmethod


class Resampling(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the class with the data."""
        pass

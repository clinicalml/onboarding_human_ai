from abc import ABC, abstractmethod


class BaseTeacher(ABC):
    """Abstract method for teacher"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """this function should fit the model and be enough to evaluate the model"""
        pass

    @abstractmethod
    def get_defer_preds(self, data_x, ai_info=None):
        """ "
        Args:
            data_x: 2d numpy array of the features
            ai_info: 2d numpy array of the AI information
        Returns:
            defer_preds: 1d numpy array of the defer predictions
        """
        pass

    @abstractmethod
    def get_region_labels(self, data_x, ai_info=None):
        """ "
        Args:
            data_x: 2d numpy array of the features
            ai_info: 2d numpy array of the AI information
        Returns:
            data_region_labeling: 1d numpy array of the data region labeling
        """
        pass

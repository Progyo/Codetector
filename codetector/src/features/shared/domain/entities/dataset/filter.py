from abc import ABC, abstractmethod
from ..samples import Sample

class Filter(ABC):
    """
    Abstract class representing all the filters in the framework.
    """

    @abstractmethod
    def isAllowed(self, sample:Sample, index:int=None) -> bool:
        """
        Returns `True` if the sample passes the filter.
        index: optional parameter that represents the index of the sample in the dataset.
        """
        pass
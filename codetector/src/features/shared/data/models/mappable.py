from abc import ABC, abstractmethod
from codetector.src.features.shared.domain.entities.samples.sample import Sample

class MappableMixin(ABC):
    """
    Mixin used by sample models to indicate that they can be converted to and from a dict.
    """

    @staticmethod
    @abstractmethod 
    def fromDict(sample:dict) -> Sample:
        """
        Convert the dictionary representation of a sample in to a `Sample` instance.
        """
        pass

    @abstractmethod
    def toDict(self, stringify:bool=True) -> dict:
        """
        Return a dictionary representation of the sample.
        stringify: Convert all attributes into strings.
        """
        pass
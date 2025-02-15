from abc import ABC, abstractmethod

from codetector.src.features.shared.domain.entities.samples.sample import Sample


class BaseModel(ABC):
    """
    Abstract class representing all models in the framework.
    """

    def __str__(self) -> str:
        return self.getModelName()
    

    def initialize(self, load:bool=False) -> None:
        """
        Initialize the model.
        """
        if load:
            self.load()

    @abstractmethod
    def getTag(self) -> str:
        """
        Return the tag of the model.
        """
        pass

    @abstractmethod
    def getModelName(self) -> str:
        """
        Return the model name of the underlying LLM.
        """
        pass

    @abstractmethod
    def isLoaded(self) -> bool:
        """
        Return whether the model is loaded or not.
        """
        pass

    @abstractmethod
    def load(self, outputHiddenStates:bool=False) -> None:
        """
        Load the model into memory.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model.
        """
        pass

    @abstractmethod
    def useSDP(self) -> bool:
        """
        Return `True` if the model should use SDP optimization.
        """
        pass


    @abstractmethod
    def toPromptFormat(self, sample:Sample) -> str:
        """
        Convert the sample prompt into the correct format.
        """
        pass

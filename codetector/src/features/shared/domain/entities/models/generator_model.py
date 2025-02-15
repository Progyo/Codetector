from abc import ABC, abstractmethod
from typing import Callable
from ..samples import Sample

class GeneratorMixin(ABC):
    """
    Mixin for all models that support generation in the framework.
    """

    @abstractmethod
    def generateSingle(self, sample:Sample) -> list[Sample]:
        """
        Generate sample(s) using a single reference sample.
        """
        pass

    @abstractmethod
    def generateBatch(self, sampleslist:list[Sample]) -> list[list[Sample]]:
        """
        Generate samples using a batch of samples.
        """
        pass


    @abstractmethod
    def setTemperature(self, temperature:float) -> None:
        """
        Set the temperature of the model.
        """
        pass

    @abstractmethod
    def setTopK(self, k:int) -> None:
        """
        Set the top K value.
        """
        pass

    @abstractmethod
    def setTopP(self, p:float) -> None:
        """
        Set the top P value.
        """
        pass

    @abstractmethod
    def setMaxOutputLength(self, length:int) -> None:
        """
        Set the maximum output length of the model.
        """
        pass


    @abstractmethod
    def setGenerateCount(self, count:int) -> None:
        """
        Set the number of samples to generate per reference sample.
        """
        pass

    @abstractmethod
    def setBestOf(self, bestOf:int, heuristic:Callable[[list[Sample]],Sample]) -> None:
        """
        Set the number of samples to generate and select the best of using heuristic (callback).
        """
        pass

    @abstractmethod
    def enableDynamicSize(self, enable:bool) -> None:
        """
        Whether to enable dynamic size. Dynamic size automatically adjusts the
        maximum size of the output length according to the sample's length.
        """
        pass

    @abstractmethod
    def supportsSample(self, sample:Sample) -> bool:
        """
        Returns whether the type of sample is supported by the model.
        """
        pass
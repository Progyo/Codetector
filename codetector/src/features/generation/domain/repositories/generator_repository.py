from abc import ABC, abstractmethod
from codetector.src.core import Result, NoneResult
from codetector.src.features.shared.domain.entities import Sample, GeneratorMixin
from typing import Callable

class GeneratorRepository(ABC):
    """
    Interface for generator repository implementations.
    """

    @abstractmethod
    def initialize(self) -> NoneResult:
        """
        Initialize the generator repository.
        """
        pass


    @abstractmethod
    def generateSingle(self, sample:Sample) -> Result[list[list[Sample]]]:
        """
        Generate sample(s) from a single reference sample.
        """
        pass


    @abstractmethod
    def generateBatch(self, samples:list[Sample]) -> Result[list[list[list[Sample]]]]:
        """
        Generate samples using a batch of samples.
        """
        pass

    @abstractmethod
    def setTemperature(self, temperature:float) -> NoneResult:
        """
        Set the temperature of all the generator models.
        """
        pass

    @abstractmethod
    def setTopK(self, k: int) -> NoneResult:
        """
        Set the top k value of all the generator models.
        """
        pass

    @abstractmethod
    def setTopP(self, p: float) -> NoneResult:
        """
        Set the top p value of all the generator models.
        """
        pass

    @abstractmethod
    def setMaxOutputLength(self, length:int) -> NoneResult:
        """
        Set the max output length of all the generator models.
        """
        pass

    @abstractmethod
    def setGenerateCount(self, count:int) -> NoneResult:
        """
        Set the number of samples to generate per reference sample.
        """
        pass

    @abstractmethod
    def addGenerator(self, generator:GeneratorMixin) -> NoneResult:
        """
        Add a generator model to the list of generators being managed.
        """
        pass

    @abstractmethod
    def setBestOf(self, bestOf:int, heuristic:Callable[[list[Sample]],Sample]) -> NoneResult:
        """
        Set the number of samples to generate and select the best of using heuristic (callback).
        """
        pass

    @abstractmethod
    def enableDynamicSize(self, enable:bool) -> NoneResult:
        """
        Whether to enable dynamic size. Dynamic size automatically adjusts the
        maximum size of the output length according to the sample's length.
        """
        pass
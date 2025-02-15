from abc import ABC, abstractmethod
from codetector.src.core import Result, NoneResult
from codetector.src.features.shared.domain.entities import Sample, DetectorMixin, DetectionSample
from ..entities.detector import BaseDetector

class DetectorRepository(ABC):
    """
    Interface for detector repository implementations.
    """

    @abstractmethod
    def initialize(self) -> NoneResult:
        """
        Initialize the detector repository.
        """ 
        pass

    @abstractmethod
    def detect(self, samples:list[Sample]) -> Result[list[DetectionSample]]:
        """
        Analyze the samples using the loaded detectors and registered base models and secondary models.
        """
        pass

    @abstractmethod
    def addDetector(self, detector: BaseDetector) -> NoneResult:
        """
        Add a detector to the detection methods to be tested.
        """
        pass

    @abstractmethod
    def registerPrimaryModel(self, model:DetectorMixin) -> NoneResult:
        """
        Inform the pipeline of the primary (base) model to be used by the detectors.
        """
        pass

    @abstractmethod
    def registerSecondaryModels(self, model:DetectorMixin, secondaryModels:list[DetectorMixin]) -> NoneResult:
        """
        Inform the pipeline of the valid secondary model combinations that are compatible with the primary (base) model.
        """
        pass

    @abstractmethod
    def setMaxDetectionLength(self, length:int|None) -> NoneResult:
        """
        Set the maximum amount of tokens a detector is allowed to use for detection from a sample.
        """
        pass
from abc import ABC, abstractmethod
from codetector.src.features.shared.domain.entities import Sample, DetectorMixin, DetectionSample

class BaseDetector(ABC):
    """
    Abstract class representing all the detectors/detection methods implemented in the framework.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the detector.
        """
        pass


    @abstractmethod
    def detect(self, samples:list[Sample]) -> list[float]:
        """
        Analyze the list of samples.
        """
        pass


    @abstractmethod
    def setPrimaryModel(self, model:DetectorMixin) -> None:
        """
        Set the primary (base) model used in the detector.
        """
        pass

    @abstractmethod
    def getTag(self) -> str:
        """
        Return the tag of the detector.
        """
        pass
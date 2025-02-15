from abc import ABC, abstractmethod
from codetector.src.features.shared.domain.entities import Sample, DetectionSample, BaseModel, DetectorMixin

class TwoModelDetectorMixin(ABC):
    """
    Mixin for all detectors that use two models for detection in the framework.
    """
    
    def __init__(self, keepBothModelsLoaded:bool=False):
        pass
        

    @abstractmethod
    def setSecondaryModel(self, secondaryModel:DetectorMixin) -> None:
        """
        Set the secondary model used by the detector.
        """
        pass
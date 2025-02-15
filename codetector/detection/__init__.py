#Expose repository under codetector.detection
from ..src.features.detection.data.repositories.detector_implementation import DetectorRepositoryImplementation as DetectorManager
from ..src.features.detection.domain.entities.detector import BaseDetector as SingleModelDetector
from ..src.features.detection.domain.entities.two_model_detector import TwoModelDetectorMixin
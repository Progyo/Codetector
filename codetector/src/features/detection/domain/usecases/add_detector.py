from codetector.src.core import UsecaseWithParameters,NoneResult
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities.models import DetectorMixin
from dataclasses import dataclass

@dataclass(frozen=True)
class AddDetectorParameters:
    """
    Parameters for the AddDetector use case.
    """
    detector: DetectorMixin
    """
    The detector to add.
    """

class AddDetector(UsecaseWithParameters[AddDetectorParameters, NoneResult]):
    """
    Use case that adds detectors.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:AddDetectorParameters) -> NoneResult:
        return self.repository.addDetector(params.detector)



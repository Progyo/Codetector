from codetector.src.core import UsecaseWithParameters, Result
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities import DetectorMixin, Sample, DetectionSample
from dataclasses import dataclass
from oslash import Right,Left

@dataclass(frozen=True)
class DetectParameters:
    """
    Parameters for the RegisterPrimaryModels use case.
    """
    samples: list[Sample]
    """
    The samples to analyse in the pipeline.
    Should not contain detection samples
    """
    maxLength: int|None = None
    """
    The length to set. If `None`, then no maximum length.
    """

class Detect(UsecaseWithParameters[DetectParameters, Result[list[DetectionSample]]]):
    """
    Use case that actually outputs detection samples from samples using added detectors and registered models.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:DetectParameters) -> Result[list[DetectionSample]]:
        
        result = self.repository.setMaxDetectionLength(params.maxLength)

        if isinstance(result,Left):
            return result

        return self.repository.detect(params.samples)




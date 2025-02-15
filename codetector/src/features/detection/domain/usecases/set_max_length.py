from codetector.src.core import UsecaseWithParameters,NoneResult
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities.models import DetectorMixin
from dataclasses import dataclass

@dataclass(frozen=True)
class SetMaxLengthParameters:
    """
    Parameters for the SetMaxLength use case.
    """
    length: int|None
    """
    The length to set. If `None`, then no maximum length.
    """

class SetMaxLength(UsecaseWithParameters[SetMaxLengthParameters, NoneResult]):
    """
    Use case that sets the maximum length a detector is allowed to use for detection from a sample.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:SetMaxLengthParameters) -> NoneResult:
        return self.repository.setMaxDetectionLength(params.length)




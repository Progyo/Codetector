from codetector.src.core import UsecaseWithoutParameters,NoneResult
from codetector.src.features.detection.domain.repositories import DetectorRepository

class Initialize(UsecaseWithoutParameters[NoneResult]):
    """
    Use case that initializes the detector.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository

    def __call__(self) -> NoneResult:
        return self.repository.initialize()
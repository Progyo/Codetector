from codetector.src.core import UsecaseWithParameters,NoneResult
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities.models import DetectorMixin
from dataclasses import dataclass
from oslash import Right,Left

@dataclass(frozen=True)
class RegisterPrimaryModelsParameters:
    """
    Parameters for the RegisterPrimaryModels use case.
    """
    models: list[DetectorMixin]
    """
    The models to register in the detection pipeline.
    """

class RegisterPrimaryModels(UsecaseWithParameters[RegisterPrimaryModelsParameters, NoneResult]):
    """
    Use case that registers models to the pipeline.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:RegisterPrimaryModelsParameters) -> NoneResult:
        for model in params.models:
            result = self.repository.registerPrimaryModel(model)
            if isinstance(result,Left):
                return result

        return Right(None)




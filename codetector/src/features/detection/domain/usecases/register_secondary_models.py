from codetector.src.core import UsecaseWithParameters,NoneResult
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities.models import DetectorMixin
from dataclasses import dataclass
from oslash import Right,Left

@dataclass(frozen=True)
class RegisterSecondaryModelsParameters:
    """
    Parameters for the RegisterSecondaryModels use case.
    """
    models: dict[DetectorMixin,list[DetectorMixin]]
    """
    The secondary models to register in the detection pipeline.
    Should only contain compatible models.
    """

class RegisterSecondaryModels(UsecaseWithParameters[RegisterSecondaryModelsParameters, NoneResult]):
    """
    Use case that registers models to the pipeline.
    """

    def __init__(self, repository:DetectorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:RegisterSecondaryModelsParameters) -> NoneResult:
        for model in params.models:
            result = self.repository.registerSecondaryModels(model,params.models[model])
            if isinstance(result,Left):
                return result

        return Right(None)




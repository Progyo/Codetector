from codetector.src.core import UsecaseWithParameters,NoneResult
from codetector.src.features.generation.domain.repositories import GeneratorRepository
from codetector.src.features.shared.domain.entities.models import GeneratorMixin
from dataclasses import dataclass

@dataclass(frozen=True)
class AddGeneratorParameters:
    """
    Parameters for the AddGenerator use case.
    """
    generator: GeneratorMixin
    """
    The generator to add.
    """

class AddGenerator(UsecaseWithParameters[AddGeneratorParameters, NoneResult]):
    """
    Use case that adds generators.
    """

    def __init__(self, repository:GeneratorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:AddGeneratorParameters) -> NoneResult:
        return self.repository.addGenerator(params.generator)



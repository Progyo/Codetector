from codetector.src.core import UsecaseWithoutParameters,NoneResult
from codetector.src.features.generation.domain.repositories import GeneratorRepository

class Initialize(UsecaseWithoutParameters[None]):
    """
    Use case that initializes the generator.
    """

    def __init__(self, repository:GeneratorRepository):
        super().__init__()
        self.repository = repository

    def __call__(self) -> NoneResult:
        return self.repository.initialize()
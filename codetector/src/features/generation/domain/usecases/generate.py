from codetector.src.core import UsecaseWithParameters,Failure,Result
from codetector.src.features.generation.domain.repositories import GeneratorRepository
from codetector.src.features.shared.domain.entities.samples import Sample
from dataclasses import dataclass
from typing import Callable
from oslash import Left

@dataclass(frozen=True)
class GenerateParameters:
    """
    Parameters for the Generate use case.
    """
    samples: list[Sample]
    """
    The samples to use as reference for generation.
    """
    batch: bool
    """
    Whether to batch generate the samples.
    """
    temperature: float|None = None
    """
    The temperature value to use. If `None` do not update.
    """
    topK: int|None = None
    """
    The top k value to use. If `None` do not update.
    """
    topP: float|None = None
    """
    The top p value to use. If `None` do not update.
    """
    bestOf: tuple[int,Callable|None]|None = None
    """
    The number of samples to select the best from. If `None` do not update.
    """
    generateCount: int|None = None
    """
    The number of samples to generate per reference sample. If `None` do not update.
    """
    dynamicSize: bool|None = None
    """
    Whether to use dynamic sizing. If `None` do not update.
    """


class Generate(UsecaseWithParameters[GenerateParameters, Result[list[list[Sample]]|list[list[list[Sample]]]]]):
    """
    Use case that generates new samples.
    """

    def __init__(self, repository:GeneratorRepository):
        super().__init__()
        self.repository = repository


    def __call__(self, params:GenerateParameters) -> Result[list[list[Sample]]|list[list[list[Sample]]]]:

        if params.temperature != None:
            result = self.repository.setTemperature(params.temperature)
            if (isinstance(result,Left)):
                return result

        if params.topK != None:
            result = self.repository.setTopK(params.topK)
            if (isinstance(result,Left)):
                return result

        if params.topP != None:
            result = self.repository.setTopP(params.topP)
            if (isinstance(result,Left)):
                return result

        if params.generateCount != None:
            result = self.repository.setGenerateCount(params.generateCount)
            if (isinstance(result,Left)):
                return result

        if params.dynamicSize != None:
            result = self.repository.enableDynamicSize(params.dynamicSize)
            if (isinstance(result,Left)):
                return result


        if params.bestOf != None:
            result = self.repository.setBestOf(params.bestOf[0],heuristic=params.bestOf[1])
            if (isinstance(result,Left)):
                return result

        if params.batch:
            return self.repository.generateBatch(params.samples)

        if len(params.samples) == 1:
            return self.repository.generateSingle(params.samples[0])
        else:
            return Left(Failure(f'Incorrect sample count {len(params.samples)} passed when generating single', -1))



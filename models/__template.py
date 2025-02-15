from typing import Callable
from codetector.models import BaseModel, GeneratorMixin, DetectorMixin
from codetector.samples.abstract import CodeSample, Sample


class TemplateGenerator(BaseModel, GeneratorMixin):

    __MODEL_MAX_OUTPUT_LENGTH = 1024

    def __init__(self):
        super().__init__()
        self.__topP : float = 0.95
        self.__topK : int = None
        self.__temperature : float = 0.97
        self.__outputLength : int = 256
        self.__outputCount : int = 1

        self.__bestOf : int = 1
        self.__bestOfHeuristic : Callable[[list[Sample]],Sample] = None

        self.__dynamicSize : bool = False

    def initialize(self, load:bool=False) -> None:
        super().initialize(load=load)
        #Do something if necessary


    def getTag(self) -> str:
        raise NotImplementedError

    def getModelName(self) -> str:
        raise NotImplementedError

    def isLoaded(self) -> bool:
        return hasattr(self,'__model') and hasattr(self,'__tokenizer')

    def load(self, outputHiddenStates:bool=False) -> None:
        if not self.isLoaded():
            pass

    def unload(self) -> None:
        raise NotImplementedError

    def useSDP(self) -> bool:
        raise NotImplementedError

    def generateSingle(self, sample:Sample) -> list:
        raise NotImplementedError

    def generateBatch(self, sampleslist:list[Sample]) -> list[list[Sample]]:
        raise NotImplementedError

    def setTemperature(self, temperature:float) -> None:
        self.__temperature = temperature

    def setTopK(self, k:int) -> None:
        self.__topK = k

    def setTopP(self, p:float) -> None:
        self.__topP = p

    def setMaxOutputLength(self, length:int) -> None:
        self.__outputLength = min(TemplateGenerator.__MODEL_MAX_OUTPUT_LENGTH, length)

    def setGenerateCount(self, count:int) -> None:
        self.__outputCount = count

    def setBestOf(self, bestOf:int, heuristic:Callable[[list[Sample]],Sample]) -> None:
        self.__bestOf = bestOf
        self.__bestOfHeuristic = heuristic

    def enableDynamicSize(self, enable:bool) -> None:
        self.__dynamicSize = enable

    def supportsSample(self, sample:Sample) -> bool:
        raise NotImplementedError
    


class TemplateDetector(BaseModel,DetectorMixin):

    def initialize(self, load:bool=False) -> None:
        super().initialize(load=load)
        #Do something is necessary

    def getTag(self) -> str:
        raise NotImplementedError

    def getModelName(self) -> str:
        raise NotImplementedError

    def isLoaded(self) -> bool:
        return hasattr(self,'__model') and hasattr(self,'__tokenizer')

    def load(self, outputHiddenStates:bool=False) -> None:
        if not self.isLoaded():
            pass

    def unload(self) -> None:
        raise NotImplementedError

    def useSDP(self) -> bool:
        raise NotImplementedError

    #### Detection methods

    def getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor, torch.FloatTensor]
        pass

    def getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]: #[torch.FloatTensor, torch.Tensor, torch.FloatTensor]
        pass

    def getLoss(self, sample:Sample) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor]
        pass

    def getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]: #[torch.FloatTensor, list, torch.FloatTensor]
        pass

    def getPadTokenId(self) -> int:
        pass

    def isLargeModel(self) -> bool:
        pass
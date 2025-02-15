from abc import ABC, abstractmethod
from ..samples import Sample

class DetectorMixin(ABC):
    """
    Mixin for all models that support detection in the framework.
    """

    @abstractmethod
    def getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor, torch.FloatTensor]
        """
        Return the logits vector given the input sample.
        """
        pass

    @abstractmethod
    def getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]: #[torch.FloatTensor, torch.Tensor, torch.FloatTensor]
        """
        Return list of logits vectors given the input samples.
        """
        pass

    @abstractmethod
    def getLoss(self, sample:Sample) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor]
        """
        Return the loss given the input sample.
        """
        pass

    @abstractmethod
    def getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]: #[torch.FloatTensor, list, torch.FloatTensor]
        """
        Return list of losses given the input samples.
        """
        pass

    @abstractmethod
    def getPadTokenId(self) -> int:
        """
        Return the pad token id for this model.
        """
        pass

    @abstractmethod
    def isLargeModel(self) -> bool:
        """
        Return `True` if only this model should be loaded in memory.
        """
        pass
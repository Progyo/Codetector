from codetector.detection import SingleModelDetector
from codetector.models import DetectorMixin, BaseModel
from codetector.samples.abstract import Sample, DetectionSample

class EntropyDetector(SingleModelDetector):
    """
    Entropy implementation.
    """

    def __init__(self):
        super().__init__()
        self.__model : BaseModel|DetectorMixin = None
        """
        The underlying model used to generate the logits.
        """

    def initialize(self) -> None:
        try:
            import torch.nn.functional
            self.__F = torch.nn.functional
        except ModuleNotFoundError:
            raise Exception('Detector requires PyTorch!')

    def __detect(self, sample:Sample) -> float:
        #Code altered from: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L324
    
        logits, labels, attention_mask, hidden_states = self.__model.getLogits(sample)
        # logits = logits[:,:-1]

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        if attention_mask != None:
            attention_mask = attention_mask[..., 1:].contiguous()

        #logits.shape = torch.Size([1, len(sample), 51200])
        
        #Results in a tensor where the last dimensions is equivalent to p(X_i=w_i|X_{1:i-1}) * log(p(X_i=w_i|X_{1:i-1}))
        neg_entropy = self.__F.softmax(logits, dim=-1) * self.__F.log_softmax(logits, dim=-1)         

        return -neg_entropy.sum(-1).mean().item()

    #This could be accelerated by using getLogitsBatch
    def detect(self, samples:list[Sample]) -> list[float]:
        if self.__model == None:
            raise Exception('No model set in entropy detector!')
        
        toReturn : list[float] = []
        for sample in samples:
            toReturn.append(self.__detect(sample))

        return toReturn

    def setPrimaryModel(self, model:DetectorMixin) -> None:
        self.__model = model

    def getTag(self) -> str:
        return 'entropy'
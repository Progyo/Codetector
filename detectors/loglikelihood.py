from codetector.detection import SingleModelDetector
from codetector.models import DetectorMixin, BaseModel
from codetector.samples.abstract import Sample, DetectionSample

class LogLikelihoodDetector(SingleModelDetector):

    def __init__(self):
        super().__init__()
        self.__model : BaseModel|DetectorMixin = None
        """
        The underlying model used to generate the log-likelihood value.
        """

    def initialize(self) -> None:
        pass

    def detect(self, samples:list[Sample]) -> list[float]:
        if self.__model == None:
            raise Exception('No model set in log-likelihood detector!')
        
        #Code altered from: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L266
        toReturn : list[float] = []
        for sample in samples:
            loss, labels, attention_mask = self.__model.getLoss(sample)
            toReturn.append(-loss.item())

        return toReturn

    def setPrimaryModel(self, model:DetectorMixin) -> None:
        self.__model = model

    def getTag(self) -> str:
        return 'loglikelihood'
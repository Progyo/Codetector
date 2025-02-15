from codetector.detection import SingleModelDetector
from codetector.models import DetectorMixin, BaseModel
from codetector.samples.abstract import Sample, DetectionSample

class RankDetector(SingleModelDetector):

    def __init__(self):
        super().__init__()
        self.__model : BaseModel|DetectorMixin = None
        """
        The underlying model used to generate the logits.
        """

    def initialize(self) -> None:
        try:
            import torch
            self.__torch = torch
        except ModuleNotFoundError:
            raise Exception('Detector requires PyTorch!')

    def __detect(self, sample:Sample) -> float:
        #Code altered from: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298
        logits, labels, attention_mask, hidden_states = self.__model.getLogits(sample)
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        if attention_mask != None:
            attention_mask = attention_mask[..., 1:].contiguous()

        # logits = logits[:,:-1]
        # labels = labels[:,1:]

        # Does nothing since order is preserved
        # logits = torch.softmax(logits,dim=-1)

        #logits.shape = torch.Size([1, len(sample), 51200])

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        #ranks = list of ranks for each token
        #timesteps = list of indexes of each token in the input


        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == self.__torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank

        return ranks.float().mean().item()

    def detect(self, samples:list[Sample]) -> list[float]:
        if self.__model == None:
            raise Exception('No model set in rank detector!')
        
        toReturn : list[float] = []
        for sample in samples:
            toReturn.append(self.__detect(sample))

        return toReturn
        

    def setPrimaryModel(self, model:DetectorMixin) -> None:
        self.__model = model

    def getTag(self) -> str:
        return 'rank'
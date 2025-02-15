from codetector.detection import SingleModelDetector, TwoModelDetectorMixin
from codetector.models import DetectorMixin, BaseModel
from codetector.samples.abstract import Sample, DetectionSample
import numpy as np

#Many implementation details
#from: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py
class FastDetectGPT(SingleModelDetector,TwoModelDetectorMixin):
    """
    Implementation of Fast-DetectGPT.
    Adapted from: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py
    """


    def __init__(self):
        super().__init__()
        self.__model : BaseModel|DetectorMixin = None
        """
        The underlying model used to generate the logits.
        """

        self.__secondaryModel : BaseModel|DetectorMixin = None
        """
        The second model used by the detection method.
        """

    def initialize(self) -> None:
        try:
            import torch
            self.__torch = torch
        except ModuleNotFoundError:
            raise Exception('Detector requires PyTorch!')


    #Is apparently just as accurate as calculating 10k samples but 10% faster
    def __get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = self.__torch.log_softmax(logits_score, dim=-1)
        probs_ref = self.__torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * self.__torch.square(lprobs_score)).sum(dim=-1) - self.__torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    # This can be done much more efficiently at the cost of using more memory
    # by generating all the logits per model at once, then unloading and repeating with other model
    def __detect(self, sample:Sample) -> float:


        
        logits_ref, labels_ref, attention_mask, hidden_states = self.__secondaryModel.getLogits(sample)
        logits_ref = logits_ref[..., :-1, :].contiguous()
        labels_ref = labels_ref[..., 1:].contiguous()
        if attention_mask != None:
            attention_mask = attention_mask[..., 1:].contiguous()

        if self.__secondaryModel != self.__model:

            if self.__secondaryModel.isLargeModel() or self.__model.isLargeModel():
                self.__secondaryModel.unload()

            logits_scoring, labels_scoring, attention_mask, hidden_states = self.__model.getLogits(sample)
            logits_scoring = logits_scoring[..., :-1, :].contiguous()
            labels_scoring = labels_scoring[..., 1:].contiguous()

            if self.__secondaryModel.isLargeModel() or self.__model.isLargeModel():
                self.__model.unload()


            try:
                if not self.__torch.all(labels_ref == labels_scoring):#, "Tokenizer is mismatch."
                    raise Exception("Tokenizer mismatch!")
            except RuntimeError:
                raise Exception("Tokenizer mismatch!")
        else:
            logits_scoring, labels_scoring = logits_ref, labels_ref

        return self.__get_sampling_discrepancy_analytic(logits_ref, logits_scoring, labels_scoring)


    def detect(self, samples:list[Sample]) -> list[float]:
        if self.__model == None or self.__secondaryModel == None:
            raise Exception('Model not set in binoculars detector!')
        
        toReturn : list[float] = []
        for sample in samples:
            toReturn.append(self.__detect(sample))

        return toReturn

    def setPrimaryModel(self, model:DetectorMixin) -> None:
        self.__model = model

    def setSecondaryModel(self, secondaryModel:DetectorMixin):
        self.__secondaryModel = secondaryModel


    def getTag(self) -> str:
        return 'fastdetectgpt'
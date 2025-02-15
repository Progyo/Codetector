from codetector.detection import SingleModelDetector, TwoModelDetectorMixin
from codetector.models import DetectorMixin, BaseModel
from codetector.samples.abstract import Sample, DetectionSample
import numpy as np

class BinocularsDetector(SingleModelDetector,TwoModelDetectorMixin):
    """
    Implementation of Binoculars.
    Paper: arXiv:2401.12070
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
            self.__ce_loss_fn : torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="none")
            self.__softmax_fn : torch.nn.Softmax = torch.nn.Softmax(dim=-1)
        except ModuleNotFoundError:
            raise Exception('Detector requires PyTorch!')






    #Called log perplexity in the paper
    #From: https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py
    def __perplexity(self, logits,
                    attention_mask,
                    input_ids,
                    temperature: float = 1.0):
        
        
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = input_ids[..., 1:].contiguous()
        
        if attention_mask != None:
            shifted_attention_mask = attention_mask[..., 1:].contiguous()


            ppl = (self.__ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        else:
            ppl = (self.__ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)).sum(1) / shifted_logits.shape[1]
            ppl = ppl.to("cpu").float().numpy()    

        return ppl

    #Called log cross perplexity
    #From: https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py
    def __entropy(self, p_logit,
                q_logits,
                input_ids,
                pad_token_id: int,
                median: bool = False,
                sample_p: bool = False,
                temperature: float = 1.0):
        
        ###Added
        p_logits = p_logits[..., :-1, :].contiguous()
        q_logits = q_logits[..., :-1, :].contiguous()
        input_ids = input_ids[..., 1:].contiguous()
        ###

        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores, q_scores = p_logits / temperature, q_logits / temperature


        p_proba = self.__softmax_fn(p_scores).view(-1, vocab_size)

        if sample_p:
            p_proba = self.__torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

        q_scores = q_scores.view(-1, vocab_size)

        if q_scores.nelement() == 0 or p_proba.nelement() == 0:
            raise Exception('Empty Tensor')

        ce = self.__ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (input_ids != pad_token_id).type(self.__torch.uint8)

        if median:
            ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
            agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
        else:
            agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

        return agg_ce


    # This can be done much more efficiently at the cost of using more memory
    # by generating all the logits per model at once, then unloading and repeating with other model
    def __detect(self, sample:Sample) -> float:
         #Performer = self.baseModel
        #Observer = self.perturbationModel

        
        logits_perf, labels_perf, attention_mask_perf, hidden_states = self.__model.getLogits(sample)

        if self.__secondaryModel != self.__model:
            
            if self.__secondaryModel.isLargeModel() or self.__model.isLargeModel():
                self.__model.unload()

            logits_obs, labels_obs, attention_mask_obs, hidden_states = self.perturbationModel.getLogits(sample)

            if self.__secondaryModel.isLargeModel() or self.__model.isLargeModel():
                self.__secondaryModel.unload()

            try:
                if not self.__torch.all(labels_obs == labels_perf):#, "Tokenizer is mismatch."
                    raise Exception("Tokenizer mismatch!")
                    
            except RuntimeError:
                raise Exception("Tokenizer mismatch!")
                
        else:
            logits_obs, labels_obs = logits_perf, labels_perf


        if self.__model.getPadTokenId() != self.__secondaryModel.getPadTokenId():
            raise Exception('Pad token mismatch!')

        ppl = self.__perplexity(logits_perf,attention_mask_perf,labels_perf)

        try:
            x_ppl = self.__entropy(logits_obs,
                                logits_perf,
                                labels_perf,
                                self.__model.getPadTokenId())

        except Exception:
            return None

        return (ppl/x_ppl).item()


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
        return 'binoculars'
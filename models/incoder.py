import gc
from math import ceil
from types import NoneType
from typing import Callable
import time

from codetector.models import BaseModel, GeneratorMixin, DetectorMixin
from codetector.samples.abstract import Sample
from codetector.samples import CodeSample

class Incoder_1B(BaseModel, GeneratorMixin, DetectorMixin):
    """
    Implementation of the Incoder 1B LLM.
    """
    __MODEL_MAX_OUTPUT_LENGTH = 2048
    
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
        
        #Do something if necessary
        try:
            import torch
            self.__torch = torch

        except ModuleNotFoundError:
            raise Exception('Model requires PyTorch!')


        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            self.__AutoModelForCausalLM = AutoModelForCausalLM
            self.__AutoTokenizer = AutoTokenizer
            self.__BitsAndBytesConfig = BitsAndBytesConfig
        except ModuleNotFoundError:
            raise Exception('Model requires Huggingface Transformers!')
        
        super().initialize(load=load)

    def getTag(self) -> str:
        return 'incoder-1b'

    def getModelName(self) -> str:
        return 'incoder-1b-1e0e77a92e7a2a30c95da8d24d818122cfc37f7c'

    def isLoaded(self) -> bool:
        return hasattr(self,f'_{self.__class__.__name__}__model') and hasattr(self,f'_{self.__class__.__name__}__tokenizer')

    def load(self, outputHiddenStates:bool=False) -> None:
        if not self.isLoaded():
            model_name_or_path = 'facebook/incoder-1B'
            rev = '1e0e77a92e7a2a30c95da8d24d818122cfc37f7c'

            self.__tokenizer = self.__AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True, revision=rev)

            # bnb_config =  BitsAndBytesConfig(
            #                                 load_in_4bit=True,
            #                                 bnb_4bit_use_double_quant=True,
            #                                 bnb_4bit_quant_type="nf4",
            #                                 bnb_4bit_compute_dtype=self.__torch.float16
            #                                 )

            self.__model = self.__AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                trust_remote_code=True,
                                                                device_map='auto',#'cuda',
                                                                torch_dtype=self.__torch.float16,
                                                                # quantization_config=bnb_config,
                                                                output_hidden_states=outputHiddenStates,
                                                                revision=rev,)
            self.__model = self.__model.eval()

            # self.__tokenizer.bos_token_id = self.__model.config.bos_token_id
            # self.__tokenizer.eos_token_id = self.__model.config.eos_token_id
            # self.__tokenizer.pad_token_id = self.__model.config.eos_token_id
            self.__tokenizer.pad_token_id = self.__tokenizer.bos_token_id
            self.__tokenizer.padding_side = 'left'

    def unload(self) -> None:
        if self.isLoaded():
            del self.__model
            del self.__tokenizer

            gc.collect()
            self.__torch.cuda.empty_cache()
            try:
                self.__torch.mps.empty_cache()
            except RuntimeError:
                pass

    def useSDP(self) -> bool:
        return True
    
    def toPromptFormat(self, sample:CodeSample):
        return '{}\n{}\n'.format(sample.toComment(f' language: {sample.getPL()}'),
                                sample.toComment(' '+sample.prompt),)

    def generateSingle(self, sample:CodeSample) -> list[Sample]:
        self.load()

        output : list[Sample] = []

        prompt = self.toPromptFormat(sample)
        inputs = self.__tokenizer.encode(prompt, return_tensors="pt").to(self.__model.device)

        outputLength = self.__outputLength

        if self.__dynamicSize:
            outputLength = max(len(self.__tokenizer.encode(sample.content, return_tensors="pt")) * 1.25, 256)

        # attention_mask = self.__getAttentionMask(inputs)

        with self.__torch.inference_mode():
            #Not the most efficient way of doing this
            for i in range(self.__outputCount):
                tempOutputs : list[str] = []
                for j in range(self.__bestOf):
                    tempOutput = self.__model.generate(inputs,
                                                    temperature=self.__temperature,
                                                    do_sample=True,
                                                    top_k=self.__topK,
                                                    top_p=self.__topP,
                                                    # attention_mask=attention_mask,
                                                    pad_token_id=self.__tokenizer.pad_token_id,
                                                    max_new_tokens=max(min(self.__MODEL_MAX_OUTPUT_LENGTH - len(inputs), outputLength),0))
                    tempOutputs.append(self.__tokenizer.decode(tempOutput[0],skip_special_tokens=True))

                best = tempOutputs[0]
                if self.__bestOf > 1:
                    best = self.__bestOfHeuristic(tempOutputs)
                if best != None:
                    output.append(self.__toCodeSample(sample,best[len(prompt):]))
                else:
                    output.append(None)

        return output

    def generateBatch(self, samples:list[Sample]) -> list[list[Sample]]:
        self.load()

        batchSize = 4
        batches = ceil(len(samples) / batchSize) #self.__batchSize

        output : list[list[Sample]] = []

        for i in range(batches):
            output += self.__generateBatch(samples[i*batchSize:(i+1)*batchSize])
            self.__torch.cuda.empty_cache()
            try:
                self.__torch.mps.empty_cache()
            except RuntimeError:
                pass
        return output

    def __generateBatch(self, samples:list[CodeSample]) -> list[list[Sample]]:
        #From: https://huggingface.co/tiiuae/falcon-40b/discussions/50
        prompts : list[str] = list(map(lambda x:self.toPromptFormat(x),samples))
        input_tokens = self.__tokenizer.batch_encode_plus(
        prompts,
        padding='longest',
        return_tensors="pt",)
        for t in input_tokens:
            if self.__torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.__model.device)

        # attention_mask = self.__getAttentionMask(input_tokens.input_ids)
        # if attention_mask != None:
        #     input_tokens['attention_mask'] = attention_mask
        input_tokens.pop("token_type_ids")

        outputLength = self.__outputLength

        if self.__dynamicSize:
            outputLength = max(len(self.__tokenizer.encode(samples[-1].content, return_tensors="pt")) * 1.25, 256)

        output : list[Sample] = [list([None]*self.__outputCount) for i in range(len(samples))]
        
        with self.__torch.inference_mode():
            # Generate outputCount many versions of the same sample
            for i in range(self.__outputCount):

                tempoutput = self.__model.generate(**input_tokens,
                                                temperature=self.__temperature,
                                                do_sample=True,
                                                top_k=self.__topK,
                                                top_p=self.__topP,
                                                pad_token_id=self.__tokenizer.pad_token_id,
                                                max_new_tokens=max(1,min(self.__MODEL_MAX_OUTPUT_LENGTH - input_tokens.input_ids.shape[1] , outputLength)))
                
                tempoutput = self.__tokenizer.batch_decode(tempoutput,skip_special_tokens=True)

                for j in range(len(tempoutput)):
                    output[j][i] = self.__toCodeSample(samples[j],tempoutput[j][len(prompts[j]):])#tempoutput[j][len(samples[j]):]

        # Flatten output
        # output = [sample for nestedlist in output for sample in nestedlist]

        return output


    def __getAttentionMask(self, input_ids):
        """
        Return the attention mask if supported.
        Else return `None`.
        """
        return input_ids.ne(self.__tokenizer.pad_token_id).to(self.__model.device)


    def setTemperature(self, temperature:float) -> None:
        self.__temperature = temperature

    def setTopK(self, k:int) -> None:
        self.__topK = k

    def setTopP(self, p:float) -> None:
        self.__topP = p

    def setMaxOutputLength(self, length:int) -> None:
        self.__outputLength = min(Incoder_1B.__MODEL_MAX_OUTPUT_LENGTH, length)

    def setGenerateCount(self, count:int) -> None:
        self.__outputCount = count

    def setBestOf(self, bestOf:int, heuristic:Callable[[list[Sample]],Sample]) -> None:
        self.__bestOf = bestOf
        self.__bestOfHeuristic = heuristic

    def enableDynamicSize(self, enable:bool) -> None:
        self.__dynamicSize = enable

    def supportsSample(self, sample:Sample) -> bool:
        return issubclass(sample.__class__,CodeSample)
    

    def __toCodeSample(self, codesample:CodeSample, generated:str) -> CodeSample:
        """
        Create a new code sample from the one used to generate the code.
        """

        if isinstance(codesample,NoneType) or isinstance(generated,NoneType):
            return None


        #Use the fromLanguage factory to correctly create the right instance of code sample
        newCodeSample = CodeSample.fromLanguage(codesample.getPL())(generated,
                   codesample.prompt,
                   codesample.originalPrompt,
                   self.getTag(),
                   codesample.datasetTag,
                   timestamp=int(time.time()),
                   topK=self.__topK,
                   topP=self.__topP,
                   temperature=self.__temperature)
        
        return newCodeSample
    




    #### Detection methods

    def __getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple:
        inputs = self.__tokenizer(sample.content, return_tensors='pt').to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        output = self.__model(**inputs,labels=labels)

        attention_mask = None
        if hasattr(inputs,'attention_mask'):
            attention_mask = inputs.attention_mask

        hiddenStates = output.hidden_states if getHiddenStates else None

        return output.logits, labels, attention_mask, hiddenStates

    def getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLogits(sample,getHiddenStates=getHiddenStates)
        else:
            return self.__getLogits(sample,getHiddenStates=getHiddenStates)

    def __getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(list(map(lambda x:x.content,samples)), return_tensors='pt',padding=padding).to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        outputs = self.__model(**inputs,labels=labels)
        
        output: list[tuple[self.__torch.FloatTensor, list]] = []
        for i in range(0,len(samples)):
            
            logitsI = outputs.logits[i]
            labelsI = labels[i]
            attention_mask = None
            if hasattr(inputs,'attention_mask'):
                attention_mask = inputs.attention_mask[i]

            output.append(tuple([logitsI,labelsI,attention_mask]))

        return output

    def getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]: #[torch.FloatTensor, torch.Tensor, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)
        
        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLogitsBatch(samples,padding=padding)
        else:
            return self.__getLogitsBatch(samples,padding=padding)

    def __getLoss(self, sample:Sample) -> tuple:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(sample.content, return_tensors='pt').to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        output = self.__model(**inputs,labels=labels)

        attention_mask = None
        if hasattr(inputs,'attention_mask'):
            attention_mask = inputs.attention_mask

        return output.loss, labels, attention_mask

    def getLoss(self, sample:Sample) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor]

        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLoss(sample)
        else:
            return self.__getLoss(sample)

    def __getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(list(map(lambda x:x.content, samples)), return_tensors='pt',padding=padding).to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        outputs = self.__model(**inputs,labels=labels)

        output: list[tuple[self.__torch.FloatTensor, list]] = []
        for i in range(0,len(samples)):
            
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[i][..., :-1, :].contiguous()
            shift_labels = labels[i][..., 1:].contiguous()
            attention_mask = None
            if hasattr(inputs,'attention_mask'):
                attention_mask = inputs.attention_mask[i].contiguous()
            # Flatten the tokens
            loss_fct = self.__torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            output.append(tuple([loss,labels[i],attention_mask])) #outputs.logits[i],labels[i],
        return output

    def getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]: #[torch.FloatTensor, list, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLossBatch(samples,padding=padding)
        else:
            return self.__getLossBatch(samples,padding=padding)

    def getPadTokenId(self) -> int:
        if self.__tokenizer != None:
            return self.__tokenizer.pad_token_id

    def isLargeModel(self) -> bool:
        return False
    








class Incoder_6B(BaseModel, GeneratorMixin, DetectorMixin):
    """
    Implementation of the Incoder 6B LLM.
    """
    __MODEL_MAX_OUTPUT_LENGTH = 2048
    
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
        
        #Do something if necessary
        try:
            import torch
            self.__torch = torch

        except ModuleNotFoundError:
            raise Exception('Model requires PyTorch!')


        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            self.__AutoModelForCausalLM = AutoModelForCausalLM
            self.__AutoTokenizer = AutoTokenizer
            self.__BitsAndBytesConfig = BitsAndBytesConfig
        except ModuleNotFoundError:
            raise Exception('Model requires Huggingface Transformers!')
        
        super().initialize(load=load)

    def getTag(self) -> str:
        return 'incoder-6b'

    def getModelName(self) -> str:
        return 'incoder-6b-18c1a34c6f3c34f07b7a39b1ca4d07777fb9465c'

    def isLoaded(self) -> bool:
        return hasattr(self,f'_{self.__class__.__name__}__model') and hasattr(self,f'_{self.__class__.__name__}__tokenizer')

    def load(self, outputHiddenStates:bool=False) -> None:
        if not self.isLoaded():
            model_name_or_path = 'facebook/incoder-6B'
            rev = '18c1a34c6f3c34f07b7a39b1ca4d07777fb9465c'

            self.__tokenizer = self.__AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True, revision=rev)

            # bnb_config =  BitsAndBytesConfig(
            #                                 load_in_4bit=True,
            #                                 bnb_4bit_use_double_quant=True,
            #                                 bnb_4bit_quant_type="nf4",
            #                                 bnb_4bit_compute_dtype=self.__torch.float16
            #                                 )

            self.__model = self.__AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                trust_remote_code=True,
                                                                device_map='auto',#'cuda',
                                                                torch_dtype=self.__torch.float16,
                                                                # quantization_config=bnb_config,
                                                                output_hidden_states=outputHiddenStates,
                                                                revision=rev,)
            self.__model = self.__model.eval()

            # self.__tokenizer.bos_token_id = self.__model.config.bos_token_id
            # self.__tokenizer.eos_token_id = self.__model.config.eos_token_id
            self.__tokenizer.pad_token_id = self.__tokenizer.bos_token_id
            self.__tokenizer.padding_side = 'left'

    def unload(self) -> None:
        if self.isLoaded():
            del self.__model
            del self.__tokenizer

            gc.collect()
            self.__torch.cuda.empty_cache()
            try:
                self.__torch.mps.empty_cache()
            except RuntimeError:
                pass

    def useSDP(self) -> bool:
        return True
    
    def toPromptFormat(self, sample:CodeSample):
        return '{}\n{}\n'.format(sample.toComment(f' language: {sample.getPL()}'),
                                sample.toComment(' '+sample.prompt),)

    def generateSingle(self, sample:CodeSample) -> list[Sample]:
        self.load()

        output : list[Sample] = []

        prompt = self.toPromptFormat(sample)
        inputs = self.__tokenizer.encode(prompt, return_tensors="pt").to(self.__model.device)

        outputLength = self.__outputLength

        if self.__dynamicSize:
            outputLength = max(len(self.__tokenizer.encode(sample.content, return_tensors="pt")) * 1.25, 256)

        # attention_mask = self.__getAttentionMask(inputs)

        with self.__torch.inference_mode():
            #Not the most efficient way of doing this
            for i in range(self.__outputCount):
                tempOutputs : list[str] = []
                for j in range(self.__bestOf):
                    tempOutput = self.__model.generate(inputs,
                                                    temperature=self.__temperature,
                                                    do_sample=True,
                                                    top_k=self.__topK,
                                                    top_p=self.__topP,
                                                    # attention_mask=attention_mask,
                                                    pad_token_id=self.__tokenizer.pad_token_id,
                                                    max_new_tokens=max(min(self.__MODEL_MAX_OUTPUT_LENGTH - len(inputs), outputLength),0))
                    tempOutputs.append(self.__tokenizer.decode(tempOutput[0],skip_special_tokens=True))

                best = tempOutputs[0]
                if self.__bestOf > 1:
                    best = self.__bestOfHeuristic(tempOutputs)
                if best != None:
                    output.append(self.__toCodeSample(sample,best[len(prompt):]))
                else:
                    output.append(None)

        return output

    def generateBatch(self, samples:list[Sample]) -> list[list[Sample]]:
        self.load()

        batchSize = 4
        batches = ceil(len(samples) / batchSize) #self.__batchSize

        output : list[list[Sample]] = []

        for i in range(batches):
            output += self.__generateBatch(samples[i*batchSize:(i+1)*batchSize])
            self.__torch.cuda.empty_cache()
            try:
                self.__torch.mps.empty_cache()
            except RuntimeError:
                pass
        return output

    def __generateBatch(self, samples:list[CodeSample]) -> list[list[Sample]]:
        #From: https://huggingface.co/tiiuae/falcon-40b/discussions/50
        prompts : list[str] = list(map(lambda x:self.toPromptFormat(x),samples))
        input_tokens = self.__tokenizer.batch_encode_plus(
        prompts,
        padding='longest',
        return_tensors="pt",)
        for t in input_tokens:
            if self.__torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.__model.device)

        # attention_mask = self.__getAttentionMask(input_tokens.input_ids)
        # if attention_mask != None:
        #     input_tokens['attention_mask'] = attention_mask
        
        input_tokens.pop("token_type_ids")

        outputLength = self.__outputLength

        if self.__dynamicSize:
            outputLength = max(len(self.__tokenizer.encode(samples[-1].content, return_tensors="pt")) * 1.25, 256)

        output : list[Sample] = [list([None]*self.__outputCount) for i in range(len(samples))]
        
        with self.__torch.inference_mode():
            # Generate outputCount many versions of the same sample
            for i in range(self.__outputCount):

                tempoutput = self.__model.generate(**input_tokens,
                                                temperature=self.__temperature,
                                                do_sample=True,
                                                top_k=self.__topK,
                                                top_p=self.__topP,
                                                pad_token_id=self.__tokenizer.pad_token_id,
                                                max_new_tokens=max(1,min(self.__MODEL_MAX_OUTPUT_LENGTH - input_tokens.input_ids.shape[1] , outputLength)))
                
                tempoutput = self.__tokenizer.batch_decode(tempoutput,skip_special_tokens=True)

                for j in range(len(tempoutput)):
                    output[j][i] = self.__toCodeSample(samples[j],tempoutput[j][len(prompts[j]):])#tempoutput[j][len(samples[j]):]

        # Flatten output
        # output = [sample for nestedlist in output for sample in nestedlist]

        return output


    def __getAttentionMask(self, input_ids):
        """
        Return the attention mask if supported.
        Else return `None`.
        """
        return input_ids.ne(self.__tokenizer.pad_token_id).to(self.__model.device)


    def setTemperature(self, temperature:float) -> None:
        self.__temperature = temperature

    def setTopK(self, k:int) -> None:
        self.__topK = k

    def setTopP(self, p:float) -> None:
        self.__topP = p

    def setMaxOutputLength(self, length:int) -> None:
        self.__outputLength = min(Incoder_6B.__MODEL_MAX_OUTPUT_LENGTH, length)

    def setGenerateCount(self, count:int) -> None:
        self.__outputCount = count

    def setBestOf(self, bestOf:int, heuristic:Callable[[list[Sample]],Sample]) -> None:
        self.__bestOf = bestOf
        self.__bestOfHeuristic = heuristic

    def enableDynamicSize(self, enable:bool) -> None:
        self.__dynamicSize = enable

    def supportsSample(self, sample:Sample) -> bool:
        return issubclass(sample.__class__,CodeSample)
    

    def __toCodeSample(self, codesample:CodeSample, generated:str) -> CodeSample:
        """
        Create a new code sample from the one used to generate the code.
        """

        if isinstance(codesample,NoneType) or isinstance(generated,NoneType):
            return None


        #Use the fromLanguage factory to correctly create the right instance of code sample
        newCodeSample = CodeSample.fromLanguage(codesample.getPL())(generated,
                   codesample.prompt,
                   codesample.originalPrompt,
                   self.getTag(),
                   codesample.datasetTag,
                   timestamp=int(time.time()),
                   topK=self.__topK,
                   topP=self.__topP,
                   temperature=self.__temperature)
        
        return newCodeSample
    




    #### Detection methods

    def __getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple:
        inputs = self.__tokenizer(sample.content, return_tensors='pt').to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        output = self.__model(**inputs,labels=labels)

        attention_mask = None
        if hasattr(inputs,'attention_mask'):
            attention_mask = inputs.attention_mask

        hiddenStates = output.hidden_states if getHiddenStates else None

        return output.logits, labels, attention_mask, hiddenStates

    def getLogits(self, sample:Sample, getHiddenStates:bool=False) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLogits(sample,getHiddenStates=getHiddenStates)
        else:
            return self.__getLogits(sample,getHiddenStates=getHiddenStates)

    def __getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(list(map(lambda x:x.content,samples)), return_tensors='pt',padding=padding).to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        outputs = self.__model(**inputs,labels=labels)
        
        output: list[tuple[self.__torch.FloatTensor, list]] = []
        for i in range(0,len(samples)):
            
            logitsI = outputs.logits[i]
            labelsI = labels[i]
            attention_mask = None
            if hasattr(inputs,'attention_mask'):
                attention_mask = inputs.attention_mask[i]

            output.append(tuple([logitsI,labelsI,attention_mask]))

        return output

    def getLogitsBatch(self, samples:list[Sample], padding:str='do_not_pad') -> list[tuple]: #[torch.FloatTensor, torch.Tensor, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)
        
        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLogitsBatch(samples,padding=padding)
        else:
            return self.__getLogitsBatch(samples,padding=padding)

    def __getLoss(self, sample:Sample) -> tuple:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(sample.content, return_tensors='pt').to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        output = self.__model(**inputs,labels=labels)

        attention_mask = None
        if hasattr(inputs,'attention_mask'):
            attention_mask = inputs.attention_mask

        return output.loss, labels, attention_mask

    def getLoss(self, sample:Sample) -> tuple: #[torch.FloatTensor, list, torch.FloatTensor]

        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLoss(sample)
        else:
            return self.__getLoss(sample)

    def __getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]:
        #https://huggingface.co/docs/transformers/main_classes/output
        inputs = self.__tokenizer(list(map(lambda x:x.content, samples)), return_tensors='pt',padding=padding).to(self.__model.device)
        labels = inputs.input_ids

        del inputs['token_type_ids']

        outputs = self.__model(**inputs,labels=labels)

        output: list[tuple[self.__torch.FloatTensor, list]] = []
        for i in range(0,len(samples)):
            
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[i][..., :-1, :].contiguous()
            shift_labels = labels[i][..., 1:].contiguous()
            attention_mask = None
            if hasattr(inputs,'attention_mask'):
                attention_mask = inputs.attention_mask[i].contiguous()
            # Flatten the tokens
            loss_fct = self.__torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            output.append(tuple([loss,labels[i],attention_mask])) #outputs.logits[i],labels[i],
        return output

    def getLossBatch(self, samples:list[Sample], padding=None) -> list[tuple]: #[torch.FloatTensor, list, torch.FloatTensor]
        
        self.load(outputHiddenStates=True)

        if self.__torch != None:
            with self.__torch.no_grad():
                return self.__getLossBatch(samples,padding=padding)
        else:
            return self.__getLossBatch(samples,padding=padding)

    def getPadTokenId(self) -> int:
        if self.__tokenizer != None:
            return self.__tokenizer.pad_token_id

    def isLargeModel(self) -> bool:
        return True
    

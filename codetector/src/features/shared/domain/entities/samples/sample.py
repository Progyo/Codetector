from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256

@dataclass(frozen=True, init=False)
class Sample(ABC):
    """
    Immutable abstract class representing all samples in the framework.
    """

    content: str
    """
    The actual string the sample represents.
    """

    prompt: str
    """
    The prompt (or label) of the sample.
    """
    
    originalPrompt: str
    """
    The original prompt (or label) of the sample. Is useful when generating variations of the original prompt.
    """

    generatorTag: str
    """
    The tag of the generator.
    """

    datasetTag: str
    """
    The tag of the dataset.
    """

    timestamp: int
    """
    The timestamp/creation date of the sample.
    """

    topK: int|None = None
    """
    The top k value used to generate the sample.
    """

    topP: float|None = None
    """
    The top p value used to generate the sample.
    """

    temperature: float|None = None
    """
    The temperature value used to generate the sample.
    """

    def __init__(self,content:str,
                 prompt:str,
                 originalPrompt:str,
                 generatorTag:str,
                 datasetTag:str,
                 timestamp:int,
                 topK:int|None=None,
                 topP:float|None=None,
                 temperature:float|None=None,):
        super().__init__()

        super().__setattr__('content', content)
        super().__setattr__('prompt', prompt)
        super().__setattr__('originalPrompt', originalPrompt)
        super().__setattr__('generatorTag', generatorTag)
        super().__setattr__('datasetTag', datasetTag)
        super().__setattr__('timestamp', timestamp)
        super().__setattr__('topK', topK if topK != 'None' else None)
        super().__setattr__('topP', topP if topP != 'None' else None)
        super().__setattr__('temperature', temperature if temperature != 'None' else None)



    def __eq__(self, value):
        #https://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
        if isinstance(value, self.__class__):
            return self.__dict__ == value.__dict__
        return False
    

    def getHash(self,) -> str:
        """
        Return a hash representation of the sample.
        """
        return sha256(str(self).encode('utf-8')).hexdigest()

    @abstractmethod
    def toDetectionSample(self, 
                timestamp: int,
                value: float,
                detectorTag: str,
                baseModelTag: str,
                secondaryModelTag: str | None = None,
                maxLength: int | None = None,) -> 'DetectionSample': # type: ignore
        """
        Convert the sample to a detection sample.
        """
        pass

# class Sample(ABC):
#     """
#     Immutable Abstract class representing all samples in the framework.
#     """

#     def __init__(self, content:str,
#                  prompt:str,
#                  originalPrompt:str,
#                  generatorTag:str,
#                  datasetTag:str,
#                  timestamp:int,
#                  topK:int|None=None,
#                  topP:float|None=None,
#                  temperature:float|None=None,
#                  ):
#         super().__init__()


#         self.content = content
#         """
#         The actual string the sample represents.
#         """
#         self.prompt = prompt
#         """
#         The prompt (or label) of the sample.
#         """
#         self.originalPrompt = originalPrompt
#         """
#         The original prompt (or label) of the sample. Is useful when generating variations of the original prompt.
#         """
#         self.generatorTag = generatorTag
#         """
#         The tag of the generator.
#         """
#         self.datasetTag = datasetTag
#         """
#         The tag of the dataset.
#         """
#         self.timestamp = timestamp
#         """
#         The timestamp/creation date of the sample.
#         """

#         self.topK = None
#         """
#         The top k value used to generate the sample.
#         """

#         if topK != "None":
#             self.topK = topK

#         self.topP = None
#         """
#         The top p value used to generate the sample.
#         """
        
#         if topP != "None":
#             self.topP = topP

#         self.temperature = None
#         """
#         The temperature value used to generate the sample.
#         """

#         if temperature != "None":
#             self.temperature = temperature


#     def __eq__(self, value):
#         #https://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes

#         if isinstance(value, Sample):
#             return self.__dict__ == value.__dict__
#         return False
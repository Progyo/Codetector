from .sample import Sample
from abc import abstractmethod, ABC
from dataclasses import dataclass

class CodeSample(Sample, ABC):
    """
    Internal immutable abstract class representing code samples in the framework.
    """

    def __repr__(self) -> str:
        return f'PL: {self.getPL()}\nCode: {self.content}\n\n\nDataset: {self.datasetTag}\nGenerator: {self.generatorTag}\nPrompt: {self.prompt}\nOriginal Prompt: {self.originalPrompt}\nDate: {self.timestamp}\nTemperature: {self.temperature}\nTop-p: {self.topP}\nTop-k: {self.topK}'
                             
    @abstractmethod
    def toComment(self, text:str, documentation:bool=False) -> str:
        """
        Convert `text` into a comment in the format of the language.
        documentation: When `True` (if the language supports it) use a special comment format to indicate documentation of a function or variable.
        """
        pass

    @abstractmethod
    def getPL(self) -> str:
        """
        Return the programming language tag.
        """
        pass

    @abstractmethod
    def getNLPL(self) -> str:
        """
        Return the natural language representation of the programming language.
        """
        pass


#kw_only=True necessary to get __init__ to work, requires __new__ hack
# class CodeSample(_CodeSample, ABC):
#     """
#     Immutable abstract class representing code samples in the framework.
#     """

#     def __new__(mcls,
#                 content: str,
#                 prompt: str,
#                 originalPrompt: str,
#                 generatorTag: str,
#                 datasetTag: str,
#                 timestamp: int,
#                 topK: int | None = None,
#                 topP: float | None = None,
#                 temperature: float | None = None):
#         return _CodeSample(content=content,
#                            prompt=prompt,
#                            originalPrompt=originalPrompt,
#                            generatorTag=generatorTag,
#                            datasetTag=datasetTag,
#                            timestamp=timestamp,
#                            topK=topK,
#                            topP=topP,
#                            temperature=temperature)
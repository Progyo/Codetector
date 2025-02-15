from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Abstract class representing the tokenizers that can be used in the framework.
    """

    @abstractmethod
    def encode(self, text:str, stringOutput:bool=None) -> list[str]|list[int]:
        """
        Encode `text` and return a list of Strings  if `stringOutput` is `True` else list of integers.
        If `stringOutput` is `None` use default implementation.
        """
        pass

    @abstractmethod
    def decode(self, tokens:list[int]) -> str:
        """
        Decode a list of `tokens`. Must be a list of integers.
        """
        pass

    @abstractmethod
    def encodeBatch(self, texts:list[str], stringOutput:bool=None) -> list[list[str]]|list[list[int]]:
        """
        Batch encode `texts` and return a list of list of Strings  if `stringOutput` is `True` else list of list of integers.
        If `stringOutput` is `None` use default implementation.
        """
        pass


    @abstractmethod
    def decodeBatch(self, tokens:list[list[int]]) -> list[str]:
        """
        Batch decode a list of list of `tokens`. Must be a list of list of integers.
        """
        pass
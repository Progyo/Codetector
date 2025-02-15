from codetector.src.features.shared.domain.entities.tokenizer import Tokenizer

class NLTKTokenizer(Tokenizer):
    """
    Implementation of the NLTK TreebankWordTokenizer tokenizer.
    """

    def __init__(self,):
        super().__init__()

        try:
            from nltk.tokenize import TreebankWordTokenizer
        except ModuleNotFoundError as e:
            raise Exception(f'Tokenizer requires {e.name}')
        
        self.__TOKENIZER = TreebankWordTokenizer()



    def encode(self, text:str, stringOutput:bool=None) -> list[str]|list[int]:
        return self.__TOKENIZER.tokenize(text)

    def decode(self, tokens:list[int]) -> str:
        if len(tokens) > 0 and isinstance(tokens[0], int):
            raise Exception('Not supported')
        
        return ' '.join(tokens)

    def encodeBatch(self, texts:list[str], stringOutput:bool=None) -> list[list[str]]|list[list[int]]:
        return [self.encode(text) for text in texts]

    def decodeBatch(self, tokens:list[list[int]]) -> list[str]:
        return [self.decode(tokenList) for tokenList in tokens]
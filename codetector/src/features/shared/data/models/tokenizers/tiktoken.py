from codetector.src.features.shared.domain.entities.tokenizer import Tokenizer

class TikTokenTokenizer(Tokenizer):
    """
    Implementation of the TikToken tokenizer.
    """

    def __init__(self,encoding:str='o200k_base',allowedSpecial:str='all',disallowedSpecial:set=set()):
        super().__init__()

        try:
            import tiktoken
        except ModuleNotFoundError as e:
            raise Exception(f'Tokenizer requires {e.name}')
        

        self.__TOKENIZER = tiktoken.get_encoding(encoding)
        self.__allowedSpecial = allowedSpecial
        self.__disallowedSpecial = disallowedSpecial


    def encode(self, text:str, stringOutput:bool=None) -> list[str]|list[int]:
        tokens = self.__TOKENIZER.encode(text, allowed_special= self.__allowedSpecial, disallowed_special=self.__disallowedSpecial)

        if stringOutput:
            return [self.__TOKENIZER.decode([token]) for token in tokens]
        else:
            return tokens

    def decode(self, tokens:list[int]) -> str:
        return self.__TOKENIZER.decode(tokens)

    def encodeBatch(self, texts:list[str], stringOutput:bool=None) -> list[list[str]]|list[list[int]]:
        tokens = self.__TOKENIZER.encode_batch(texts, allowed_special= self.__allowedSpecial, disallowed_special=self.__disallowedSpecial)
        if stringOutput:
            return [[self.__TOKENIZER.decode([token]) for token in tokenList] for tokenList in tokens]
        else:
            return tokens

    def decodeBatch(self, tokens:list[list[int]]) -> list[str]:
        return self.__TOKENIZER.decode_batch(tokens)
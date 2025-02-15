from .code_sample_model import CodeSampleModel


class GenericSample(CodeSampleModel):
    """
    Generic code sample. Can represent any language not covered explicitly.
    """

    __PROGRAMMING_LANGUAGE = 'generic'

    __PROGRAMMING_LANGUAGE_NL = 'any programming language (or described later)'

    def toComment(self, comment:str, documentation:bool=False)->str:
        if '\n' in comment:
            return f'/*\n{comment}\n*/'
        else:
            return f'//{comment}'
        
    
    def getPL(self):
        return self.__PROGRAMMING_LANGUAGE
    
    def getNLPL(self):
        return self.__PROGRAMMING_LANGUAGE_NL
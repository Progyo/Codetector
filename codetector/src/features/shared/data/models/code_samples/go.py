from .code_sample_model import CodeSampleModel


class GoSample(CodeSampleModel):
    """
    Go code sample.
    """
    
    __PROGRAMMING_LANGUAGE = 'go'

    __PROGRAMMING_LANGUAGE_NL = 'Go'

    def toComment(self, comment:str, documentation:bool=False)->str:
        if '\n' in comment:
            return f'/*\n{comment}\n*/'
        else:
            return f'//{comment}'
        
    
    def getPL(self):
        return self.__PROGRAMMING_LANGUAGE
    
    def getNLPL(self):
        return self.__PROGRAMMING_LANGUAGE_NL
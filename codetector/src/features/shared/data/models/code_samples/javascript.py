from .code_sample_model import CodeSampleModel


class JavascriptSample(CodeSampleModel):
    """
    Javascript code sample.
    """
    
    __PROGRAMMING_LANGUAGE = 'js'

    __PROGRAMMING_LANGUAGE_NL = 'JavaScript'

    def toComment(self, comment:str, documentation:bool=False)->str:
        if '\n' in comment:
            return f'/*\n{comment}\n*/'
        else:
            return f'//{comment}'
        
    
    def getPL(self):
        return self.__PROGRAMMING_LANGUAGE
    
    def getNLPL(self):
        return self.__PROGRAMMING_LANGUAGE_NL
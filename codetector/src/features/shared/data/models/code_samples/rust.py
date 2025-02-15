from .code_sample_model import CodeSampleModel


class RustSample(CodeSampleModel):
    """
    Rust code sample.
    """

    __PROGRAMMING_LANGUAGE = 'rust'

    __PROGRAMMING_LANGUAGE_NL = 'Rust'

    def toComment(self, comment:str, documentation:bool=False)->str:
        if not documentation:
            if '\n' in comment:
                return f'/*\n{comment}\n*/'
            else:
                return f'//{comment}'
        
        else:
            return '///'+'///'.join(comment.splitlines(keepends=True))
        
    
    def getPL(self):
        return self.__PROGRAMMING_LANGUAGE
    
    def getNLPL(self):
        return self.__PROGRAMMING_LANGUAGE_NL
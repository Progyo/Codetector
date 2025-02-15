import datetime
import json
import time
from typing import Any
from codetector.dataset import HuggingFaceDataset
from codetector.samples.abstract import MappableMixin
from codetector.filters import Filter, BeforeDateFilter
from codetector.samples import CodeSample


class CodeSearchNetPythonDataset(HuggingFaceDataset):
    """
    Implementation for the huggingface CodeSearchNet dataset with python subset.
    HF Link: https://huggingface.co/datasets/code_search_net
    """


    def __init__(self,  filters : list[Filter] = [], checkPointPath = None):
        super().__init__('code_search_net','fdc6a9e39575768c27eb8a2a5f702bf846eb4759','test',
                         subset='python', filters=filters, checkPointPath=checkPointPath)

        #https://stackoverflow.com/questions/19801727/convert-datetime-to-unix-timestamp-and-convert-it-back-in-python
        #Used paper sbumisison: https://arxiv.org/abs/1909.09436
        self.__dsTime : int = time.mktime(datetime.date(2019,9,20).timetuple())

        #Filter documentation out
        #https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
        # self.__regex = re.compile(r"some-expression", re.IGNORECASE)

    def toCodeSample(self, row: dict[str, Any]) -> CodeSample | None:
        
        # #This version of the code has no comments!!!
        # code = ''.join(row['func_code_tokens'])

        prompt = f'A function called {row["func_name"].split(".")[-1]}. Functionality: {row["func_documentation_string"]}'

        code = row['whole_func_string']#.replace(prompt,'')
                                                
        sample = CodeSample.fromLanguage('python')(content=code,
                                                prompt=prompt,
                                                originalPrompt=prompt,
                                                generatorTag='human',
                                                datasetTag=self.getTag(),
                                                timestamp=self.__dsTime,
                                                topK=None,
                                                topP=None,
                                                temperature=None)
                                      
        return sample
    

    def getTag(self):
        return 'hf_codesearchnet-python'
    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
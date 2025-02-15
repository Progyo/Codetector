import datetime
import json
import time
from typing import Any
from codetector.dataset import HuggingFaceDataset
from codetector.samples.abstract import MappableMixin
from codetector.filters import Filter, BeforeDateFilter
from codetector.samples import CodeSample


class APPSDataset(HuggingFaceDataset):
    """
    Implementation for the huggingface APPS dataset.
    HF Link: https://huggingface.co/datasets/codeparrot/apps
    """


    def __init__(self,  filters : list[Filter] = [], checkPointPath = None):
        super().__init__('codeparrot/apps','21e74ddf8de1a21436da12e3e653065c5213e9d1','train',
                         subset='all', filters=filters, checkPointPath=checkPointPath)

        #https://stackoverflow.com/questions/19801727/convert-datetime-to-unix-timestamp-and-convert-it-back-in-python
        #Used this commit date: https://github.com/hendrycks/apps/commit/b064a366910559104b2b6a1f891220a45ace922e
        self.__dsTime : int = time.mktime(datetime.date(2021,4,24).timetuple())


    def toCodeSample(self, row: dict[str, Any]) -> CodeSample | None:
        
        code = json.loads(row['solutions'])[0]
        prompt = row["question"]#.split('-----Input-----')[0]

                                                
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
        return 'hf_apps'
    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
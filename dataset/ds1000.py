import datetime
import time
from codetector.dataset import HuggingFaceDataset
from codetector.samples.abstract import MappableMixin
from codetector.filters import Filter
from codetector.samples import CodeSample

class DS1000Dataset(HuggingFaceDataset):
    """
    Implementation for the huggingface DS-1000 dataset.
    """

    def __init__(self, filters : list[Filter] = [], checkPointPath = None):
        super().__init__('xlangai/DS-1000',
                         'b1d899f813ba42280d524358a541c0999c6417a4',
                         'test',
                         filters=filters,
                         checkPointPath=checkPointPath)
        #https://stackoverflow.com/questions/19801727/convert-datetime-to-unix-timestamp-and-convert-it-back-in-python
        self.__dsTime : int = time.mktime(datetime.date(2022,10,1).timetuple())


    def toSample(self, row):
        return CodeSample.fromLanguage('python')(content=row['reference_code'],
                                                prompt=row['prompt'],
                                                originalPrompt=row['prompt'],
                                                generatorTag='human',
                                                datasetTag=self.getTag(),
                                                timestamp=self.__dsTime,
                                                topK=None,
                                                topP=None,
                                                temperature=None)
    

    def getTag(self):
        return 'hf_ds1000'
    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
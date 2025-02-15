import datetime
import time
from codetector.dataset.abstract import Dataset
from codetector.dataset import DatasetBatch
from codetector.filters import Filter
from codetector.samples import CodeSample
import pandas as pd
import json



class LeetCodePostDataset(Dataset):
    """
    Implementation for custom scraped LeetCode (post August 2023) dataset.
    """

    __JSON_PATH : str = 'data/leetcode_post/samples.json'

    def __init__(self, filters : list[Filter] = [], checkpointPath = None):
        super().__init__(filters=filters, checkpointPath=checkpointPath)

        self.__hashSet : set[str] = set()
        """Prevent duplicate code samples. Not to be confused with distribution hash list!!!"""
        
        self.__samples : list[CodeSample] = []
        """
        Store the loaded samples.
        """

        self.__lastIndex = -1
        """
        The index of the last loaded sample.
        """

        self.__loadingIndex = 0
        """
        This is used to keep track of the items during loading.
        """

    def getContentType(self) -> type[CodeSample]:
        return CodeSample

    def loadDataset(self) -> None:
        with open(self.__JSON_PATH,'r') as file:
            data : list[dict[str,str|dict[str,str]]] = json.load(file)


        self.__loadingIndex = 0

        for entry in data:
            self.__samples.extend(self.__toCodeSamples(int(entry['id']), entry))


    def getTag(self) -> str:
        return 'leetcode-post'

    def loadBatch(self, size:int) -> DatasetBatch:

        samplesToReturn = self.__samples[self.__lastIndex+1:self.__lastIndex+1+size]
        self.__lastIndex += size
        done = self.__lastIndex > len(self.__samples) - 1

        return DatasetBatch(samplesToReturn, done)

    def addSample(self, sample:CodeSample) -> None:
        raise Exception('This is a source dataset. Cannot add code sample to dataset.')

    def save(self) -> None:
        raise Exception('Cannot save source dataset.')

    def getCount(self) -> int:
        return len(self.__samples)

    def toDataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(map(lambda x:x.toDict(), self.__samples)))


    def __dateRegression(self, problemId: int) -> int:
        """
        Given the problem id, calculate an approximate date of the problem.
        Return the unix epoch time.
        """

        #Using formula problemId * 0.7799434140207482 = 1.044161253570283 * day + 1827.357277250333
        # <=> day = (problemId * 0.7799434140207482 - 1827.357277250333)/1.044161253570283

        #Day here is the relative day to 'firstDay' = 2022-06-24

        firstDay = datetime.date(2022,6,24)

        day = (problemId * 0.7799434140207482 - 1827.357277250333)/1.044161253570283

        #https://stackoverflow.com/questions/6871016/adding-days-to-a-date-in-python
        actualDay = firstDay + datetime.timedelta(days=day)

        return int(time.mktime(actualDay.timetuple()))
    

    def __toCodeSamples(self, id:int, item:dict[str,str|dict[str,str]]) -> list[CodeSample]:
        
        samples : list[CodeSample] = []

        prompt = item['desc']
        date = self.__dateRegression(id)


        

        for lang, solution in item['solutions'].items():
            sample : CodeSample = CodeSample.fromLanguage(lang)(content=solution,
                                                prompt=prompt,
                                                originalPrompt=prompt,
                                                generatorTag='human',
                                                datasetTag=self.getTag(),
                                                timestamp=date,
                                                topK=None,
                                                topP=None,
                                                temperature=None)
            hashVal = sample.getHash()

            if not(hashVal in self.__hashSet) and self.passesFilters(sample,index=self.__loadingIndex):
                self.__hashSet.add(hashVal)
                samples.append(sample)
                self.__loadingIndex += 1

        return samples
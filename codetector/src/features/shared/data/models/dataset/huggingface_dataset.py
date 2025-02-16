from abc import abstractmethod
import datetime
import time
from typing import Any

import pandas as pd
import json

from codetector.src.features.shared.domain.entities.dataset import Dataset, DatasetBatch, Filter
from codetector.src.features.shared.domain.entities.samples.code_sample import CodeSample
from codetector.src.features.shared.domain.entities.samples.sample import Sample



class HuggingFaceDataset(Dataset):
    """
    Abstract class for datasets from huggingface.co
    """

    def __init__(self, datasetTag:str, revision:str, split:str, subset:str = None, filters: list[Filter] = [], checkPointPath: str = None,) -> None:
        super().__init__(filters=filters,checkpointPath=checkPointPath)
        
        #Skip initialization if loaded from checkpoint
        if self.wasLoadedFromCheckpoint():
            return


        self.__datasetTag : str = datasetTag
        """
        The tag used to load the dataset.
        """
        self.__revision : str = revision
        """
        The exact revision of the dataset.
        """
        self.__split : str = split
        """
        What split to load from the dataset.
        """
        self.__subset : str | None = subset
        """
        What subset to load from the dataset.
        """
        self.__count : int = 0
        """
        The numbers of samples contained in the unfiltered dataset.
        """
        self.__lastIndex = -1
        """
        The index of the last returned sample.
        """
        self.__left = -1
        """
        The number of samples left in the dataset.
        """
        self.__loadingIndex = 0
        """
        This is used to keep track of the items during loading.
        """
   

    def loadDataset(self) -> None:
        try:
            from datasets import load_dataset, IterableDataset
        except ModuleNotFoundError:
            raise Exception(f'{self.__class__.__name__} requires datasets package!')


        if self.__subset != None:
            self.__dataset = load_dataset(self.__datasetTag, self.__subset, revision=self.__revision, split=self.__split, keep_in_memory=False, trust_remote_code=True)
        else:
            self.__dataset  = load_dataset(self.__datasetTag, revision=self.__revision, split=self.__split, keep_in_memory=False, trust_remote_code=True)


        self.__count = len(self.__dataset)
        self.__dataset : IterableDataset = self.__dataset.to_iterable_dataset()
        self.__left = int(self.__count)


    def loadBatch(self, size:int) -> DatasetBatch:
        samplesToReturn : list[CodeSample] = []
        index = self.__lastIndex+1

        for row in self.__dataset:
            if len(samplesToReturn) >= size:
                break
            sample = self.toSample(row)
            if sample != None and self.passesFilters(sample,index=self.__loadingIndex):
                samplesToReturn.append(sample)
                self.__loadingIndex += 1
            
            index+=1
            
        self.__dataset = self.__dataset.skip(index - self.__lastIndex - 1)
        self.__left -= index - self.__lastIndex - 1
        self.__lastIndex = index-1
        done = self.__lastIndex >= self.__count - 1

        return DatasetBatch(samplesToReturn,done)

    def addSample(self, sample:CodeSample) -> None:
        raise Exception('This is a source dataset. Cannot add code sample to dataset.')

    def save(self) -> None:
        raise Exception('Cannot save source dataset.')

    def getCount(self) -> int:
        return self.__count

    def toDataframe(self) -> pd.DataFrame:
        # return pd.DataFrame(list(map(lambda x:x.toDict(), self.__samples)))
        raise Exception('Hugging Face Datasets currently do no support conversion to dataframe objects')


    @abstractmethod
    def toSample(self, row:dict[str,Any]) -> Sample | None:
        """
        Should not be called from outside.
        This is overwritten by each Hugging Face dataset due to its unique data format.
        If None returned, then skip this sample
        """
        pass
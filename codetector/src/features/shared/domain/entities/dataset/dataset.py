from abc import ABC, abstractmethod
from ..samples import Sample
from .dataset_batch import DatasetBatch
from .filter import Filter
from pathlib import Path
import pickle
from tqdm import tqdm
import pandas as pd

class Dataset(ABC):
    """
    Abstract class representing all the datasets in the framework.
    """

    def __new__(cls, *args, **kwargs):
        #checkpointPath:str=None,
        checkpointPath:str= kwargs['checkpointPath'] if 'checkpointPath' in kwargs else None
        #https://stackoverflow.com/questions/43965376/initialize-object-from-the-pickle-in-the-init-function
        if checkpointPath and Path(checkpointPath).exists():
            with open(checkpointPath,'rb') as f:
               inst = pickle.load(f)
               print(f'Loaded Checkpoint for {type(inst).__name__}')
            if not isinstance(inst, cls):
               raise TypeError('Unpickled object is not of type {}'.format(cls))
            else:
                inst.__setattr__('loadedFromCheckpoint', True)
            inst.__setattr__('checkpointPath', checkpointPath)
        else:
            inst = super(Dataset, cls).__new__(cls)#, *args, **kwargs
        return inst

    def __init__(self, filters:list[Filter]=[], checkpointPath:str=None):
        super().__init__()
        
        if not hasattr(self, 'loadedFromCheckpoint'):
            self.loadedFromCheckpoint = False

        if not hasattr(self, 'checkpointPath'):
            self.checkpointPath = checkpointPath

        # print(f'Filters: {filters}\nLoaded From Checkpoint: {self.loadedFromCheckpoint}\nCheckpoint Path: {self.checkpointPath}')

        #Don't continue initialization if loaded from checkpoint
        if self.loadedFromCheckpoint:
            return

        self.__filters = filters

    @abstractmethod
    def getContentType(self) -> type[Sample]:
        """
        Return the type of model implementation to use.
        """
        pass

    @abstractmethod
    def loadDataset(self) -> None:
        """
        Load/initialize the dataset.
        """
        pass

    @abstractmethod
    def getTag(self) -> str:
        """
        Return the tag of the dataset.
        """
        pass

    @abstractmethod
    def loadBatch(self, size:int) -> DatasetBatch:
        """
        Return a `DatasetBatch` from the dataset.
        """
        pass

    @abstractmethod
    def addSample(self, sample:Sample) -> None:
        """
        Add a sample to the dataset.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the dataset.
        """
        pass

    @abstractmethod
    def getCount(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        pass

    def passesFilters(self, sample:Sample, index:int=None) -> bool:
        """
        Return `True` if sample passes all filters.
        index: optional parameter that represents the index of the sample in the dataset.
        """

        for filter in self.__filters:
            if not filter.isAllowed(sample,index=index):
                return False
        return True
    

    def setFilters(self, filters:list[Filter]) -> None:
        """
        Set the filters of the dataset.
        """
        self.__filters = filters


    def saveCheckpoint(self) -> None:
        """
        Save current instance as pickle.
        """
        if self.checkpointPath:
            with open(self.checkpointPath,'wb') as file:
                pickle.dump(self, file)
        else:
            raise Exception('No checkpoint path but trying to save checkpoint!')

    def wasLoadedFromCheckpoint(self) -> bool:
        """
        Return whether dataset was loaded from a checkpoint.
        """
        return self.loadedFromCheckpoint
    


    def convertTo(self, datasetType:'type[Dataset]', *args, **kwargs) -> 'Dataset':
        """
        Convert the current dataset to another type.
        """

        if not issubclass(datasetType, Dataset):
            raise Exception(f'Cannot convert to {type(datasetType).__name__}')
        

        newDataset : Dataset = datasetType(*args, **kwargs)

        bar = tqdm(total=self.getCount(), desc=f'Converting {self.__class__.__name__} to {datasetType.__name__}')

        stepSize = 100

        batch = self.loadBatch(stepSize)
        while not batch.final or len(batch.samples) > 0:
            bar.update(len(batch.samples))
            for sample in batch.samples:
                newDataset.addSample(sample)
            batch = self.loadBatch(stepSize)

        newDataset.save()

        return newDataset
    

    @abstractmethod
    def toDataframe(self) -> pd.DataFrame:
        """
        Convert the dataset into a Pandas Dataframe object.
        """
        pass
import pandas as pd
from codetector.src.features.shared.data.models.mappable import MappableMixin
from codetector.src.features.shared.domain.entities.dataset.dataset import Dataset,DatasetBatch,Filter
from codetector.src.features.shared.domain.entities.samples.sample import Sample

class AggregateDataset(Dataset):
    """
    Aggregate dataset that can combine multiple datasets as one. For example, this is helpful
    when combining multiple detection datasets that were split across several instances.
    """

    def __init__(self, datasets:list[Dataset], filters : list[Filter] = [],checkpointPath:str=None):
        super().__init__(filters, checkpointPath=checkpointPath)

        #Skip initialization if loaded from checkpoint
        if self.wasLoadedFromCheckpoint():
            return

        assert len(datasets) > 0

        self.__datasets = datasets
        """
        List of datasets.
        """
        self.__count = -1
        """
        The number of samples contained in the dataset.
        """

        #Synchronize filters across datasets if set
        if len(filters) > 0:
            self.setFilters(filters)

    def getContentType(self):

        types = [dataset.getContentType() for dataset in self.__datasets]
        
        #https://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
        assert all(x == types[0] for x in types), 'Sample type mismatch in AggregateDataset!'

        return types[0]

    def loadDataset(self) -> None:
        for dataset in self.__datasets:
            dataset.loadDataset()

    def loadBatch(self, size:int) -> DatasetBatch:

        samplesToReturn = []

        currentDatasetIndex = 0
        currentDataset = self.__datasets[currentDatasetIndex]

        batch = currentDataset.loadBatch(size)

        samplesToReturn.extend(batch.samples)
        size -= len(batch.samples)

        #Empty datasets one by one
        while batch.final and size > 0 and currentDatasetIndex < len(self.__datasets)-1:
            
            currentDatasetIndex+=1
            currentDataset = self.__datasets[currentDatasetIndex]            
            batch = currentDataset.loadBatch(size)
            # print(f'Dataset: {currentDatasetIndex} Batch: {len(batch.samples)}')
            samplesToReturn.extend(batch.samples)
            size -= len(batch.samples)
            


        return DatasetBatch(samplesToReturn,batch.final)


    def addSample(self, sample:Sample|MappableMixin) -> None:
        raise Exception('Cannot write to aggregate dataset!')

    def save(self) -> None:
        raise Exception('Cannot save aggregate dataset!')
  

    def __countSamples(self) -> int:
        """
        Count the samples in the datasets.
        """

        total = 0
        for dataset in self.__datasets:
            total += dataset.getCount()
        return total

    def getCount(self) -> int:
        if self.__count == -1:
            self.__count = self.__countSamples()

        return self.__count
    

    def getTag(self):
        return '-'.join([dataset.getTag() for dataset in self.__datasets])
    

    def toDataframe(self) -> pd.DataFrame:
        return pd.concat(list(map(lambda x:x.toDataframe(),self.__datasets)))
    

    def setFilters(self, filters):
        super().setFilters(filters)
        for dataset in self.__datasets:
                dataset.setFilters(filters)

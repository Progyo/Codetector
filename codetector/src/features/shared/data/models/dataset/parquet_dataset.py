from typing import Iterator

import pandas as pd
from codetector.src.features.shared.data.models.mappable import MappableMixin
from codetector.src.features.shared.domain.entities.dataset.dataset import Dataset,DatasetBatch,Filter
from codetector.src.features.shared.domain.entities.samples.sample import Sample
from abc import abstractmethod

from pathlib import Path
import pyarrow.dataset as ds

import pyarrow.parquet as pq
from pyarrow import RecordBatch, Table, schema

class ParquetDataset(Dataset):
    """
    Class handling importing and exporting Apache Parquet datasets.
    """

    def __init__(self, folderPath:str, filters : list[Filter] = [],checkpointPath:str=None):
        super().__init__(filters, checkpointPath=checkpointPath)

        #Skip initialization if loaded from checkpoint
        if self.wasLoadedFromCheckpoint():
            return

        self.__folderPath = folderPath
        """
        The path to the folder that contains the parquet filese.
        """
        self.__count = -1
        """
        The number of samples contained in the dataset.
        """
        self.__readonly = False
        """
        If true, only read operations allowed.
        """
        self.__samples : list[Sample|MappableMixin] = []
        """
        The samples currently loaded in memory. This can either be samples waiting to be written to disk
        or samples that have been read from disk.
        """
        self.__SAVE_THRESHOLD = 100_000
        """
        The threshold to save samples to disk.
        """
        self.__nextFile = 0
        """
        The index of the next label file that needs to be loaded.
        """

        #### READ STUFF
        self.__dataset : ds.Dataset = None
        """
        The Parquet dataset (Technically PyArrow Dataset).
        """
        self.__datasetIter : Iterator[RecordBatch] = None
        """
        The iterator over the Parquet dataset.
        """

        self.__loadingIndex = 0
        """
        This is used to keep track of the items during loading.
        """


        assert issubclass(self.getContentType(),MappableMixin), f'Return type \'{self.getContentType().__name__}\' of {self.__class__.__name__}.getContentType() does not implement MappableMixin!'


    @abstractmethod
    def getContentType(self) -> type[MappableMixin]:
        """
        Return the type of model implementation to use.
        """
        pass


    def __loadFile(self) -> None:
        self.__dataset = ds.dataset(self.__folderPath,format='parquet')#,partitioning=ds.partitioning(flavor='hive')
        #Use internal batching
        self.__datasetIter = self.__dataset.to_batches(batch_size=10000)
        
        
    def loadDataset(self) -> None:
        self.__readonly = True
        self.__loadFile()
        self.__loadBatch()
        self.__nextFile += 1


    def __loadBatch(self) -> bool:
        """
        Internally loop through the PyArrow RecordBatch.
        """
        try:
            intBatch = next(self.__datasetIter)
        except StopIteration:
            return True

        sampleType = self.getContentType()

        for row in intBatch.to_pylist():
            sample = sampleType.fromDict(row)
            if self.passesFilters(sample,index=self.__loadingIndex):
                self.__samples.append(sample)
                self.__loadingIndex += 1

        return False

    def loadBatch(self, size:int) -> DatasetBatch:
        if not self.__readonly:
            raise Exception('Trying to read from Parquet dataset in write mode!')

        samplesToReturn = []

        done = False

        #Case 1: All size is contained in __samples length
        if size <= len(self.__samples):
            samplesToReturn = self.__samples[:size]
            self.__samples = self.__samples[size:]

        #Case 2: Size is not contained in __samples length
        else:
            while size > len(self.__samples) and not done:
                samplesToReturn.extend(self.__samples)
                size -= len(self.__samples)
                self.__samples = []            

                done = self.__loadBatch() and len(self.__samples) == 0
                self.__nextFile += 1

            if not done and size > 0:
                samplesToReturn.extend(self.__samples[:size])
                self.__samples = self.__samples[size:]


        return DatasetBatch(samplesToReturn, done)

    def addSample(self, sample:Sample|MappableMixin) -> None:
        if self.__readonly:
            raise Exception('Trying to write to Parquet dataset in read-only mode!')
        
        self.__samples.append(sample)

        #Increment sample count
        self.__count = self.getCount() + 1

        if len(self.__samples) >= self.__SAVE_THRESHOLD:
            self.save()

    def save(self) -> None:
        if self.__readonly:
            raise Exception('Trying to save Parquet dataset in read-only mode!')
        
        
        if len(self.__samples) == 0:
            return
        

        table : Table = Table.from_pylist(list(map(lambda x: x.toDict(stringify=False), self.__samples)))
        pq.write_table(table, f'{self.__folderPath}/data_{self.__nextFile}.parquet')

        # ds.write_dataset(
        #     table,
        #     self.__folderPath,
        #     basename_template='data_{i}.parquet',
        #     format='parquet',
        #     # partitioning=ds.partitioning(field_names=list(self.__samples[0].toDict().keys()))#,flavor='hive'
        #     partitioning_flavor='hive',
        #     existing_data_behavior='overwrite_or_ignore'
        # )

        #clear samples when saving
        self.__samples = []
        self.__nextFile += 1





    def __countSamples(self) -> int:
        """
        Count the samples in the parquet file(s).
        """
        if not self.__dataset:
            return 0

        return self.__dataset.count_rows()

    def getCount(self) -> int:
        """
        Return the number of samples in the dataset.
        `WARNING: Filters are not applied to this implementation. Entire parquet is loaded!`
        """
        if self.__count == -1:
            self.__count = self.__countSamples()

        return self.__count
    

    def toDataframe(self) -> pd.DataFrame:
        """
        Convert the dataset into a Pandas Dataframe object.
        `WARNING: Filters are not applied to this implementation. Entire parquet is loaded!`
        """
        return self.__dataset.to_table().to_pandas()
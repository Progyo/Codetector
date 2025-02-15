from codetector.dataset import XMLDataset, ParquetDataset
from codetector.filters import Filter
from codetector.samples import CodeSample
from codetector.samples.abstract import MappableMixin

class XMLGeneratedCodeDataset(XMLDataset):
    """
    Generated dataset that stores all the code samples in XML format.
    """

    def __init__(self, filters : list[Filter]= None):
        super().__init__('data/generated', filters=filters, checkpointPath='data/generated_xml.pkl')

    def preProcess(self) -> None:
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'generated'
    

class ParquetGeneratedCodeDataset(ParquetDataset):
    """
    Generated dataset that stores all the code samples in Apache Parquet format.
    """

    def __init__(self, filters : list[Filter]= None):
        super().__init__('data/generated', filters=filters, checkpointPath='data/generated_parq.pkl')

    def preProcess(self) -> None:
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'generated'
from codetector.dataset import XMLDataset, ParquetDataset
from codetector.filters import Filter
from codetector.samples import CodeDetectionSample
from codetector.samples.abstract import MappableMixin

class XMLCodeDetectionDataset(XMLDataset):
    """
    Detection dataset (Final dataset) that stores all the classified samples in XML format.
    """

    def __init__(self, filters : list[Filter]= []):
        super().__init__('data/detection', filters=filters, checkpointPath='data/detection_xml.pkl')

    def preProcess(self) -> None:
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeDetectionSample
    
    def getTag(self):
        return 'detection'
    

class ParquetCodeDetectionDataset(ParquetDataset):
    """
    Detection dataset (Final dataset) that stores all the classified samples in Apache Parquet format.
    """

    def __init__(self, filters : list[Filter]= []):
        super().__init__('data/detection', filters=filters, checkpointPath='data/detection_parq.pkl')

    def preProcess(self) -> None:
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeDetectionSample
    
    def getTag(self):
        return 'detection'
from codetector.dataset import XMLDataset, ParquetDataset
from codetector.filters import Filter
from codetector.samples import CodeSample
from codetector.samples.abstract import MappableMixin

class XMLStackOverflowPreDataset(XMLDataset):
    """
    Implementation for the Stack Overflow crawl time frame 01-Aug-2015 to 07-Apr-2016.
    File: https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z (Posts.xml)
    Date: 07-Apr-2024 11:06
    """

    def __init__(self, filters : list[Filter]= []):
        super().__init__('data/stackoverflow_pre', filters=filters, checkpointPath='data/stackoverflow_pre_xml.pkl')

    def preProcess(self) -> None:
        #This supplied version of Stack Overflow Pre has already been preprocessed from raw Posts.xml,
        #including labeling and filtering samples
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'stackoverflow-pre'
    

class ParquetStackOverflowPreDataset(ParquetDataset):
    """
    Implementation for the Stack Overflow crawl time frame 01-Aug-2015 to 07-Apr-2016.
    File: https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z (Posts.xml)
    Date: 07-Apr-2024 11:06
    """

    def __init__(self, filters : list[Filter]= []):
        super().__init__('data/stackoverflow_pre', filters=filters, checkpointPath='data/stackoverflow_pre_parq.pkl')

    def preProcess(self) -> None:
        #This supplied version of Stack Overflow Pre has already been preprocessed from raw Posts.xml,
        #including labeling and filtering samples
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'stackoverflow-pre'
    






class XMLStackOverflowPostDataset(XMLDataset):
    """
    Implementation for the Stack Overflow crawl time frame 01-Aug-2023 to 07-Apr-2024.
    File: https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z (Posts.xml)
    Date: 07-Apr-2024 11:06
    """

    def __init__(self, filters : list[Filter]= None):
        super().__init__('data/stackoverflow_post', filters=filters, checkpointPath='data/stackoverflow_post_xml.pkl')

    def preProcess(self) -> None:
        #This supplied version of Stack Overflow Post has already been preprocessed from the raw Posts.xml,
        #including labeling and filtering samples
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'stackoverflow-post'
    

class ParquetStackOverflowPostDataset(ParquetDataset):
    """
    Implementation for the Stack Overflow crawl time frame 01-Aug-2015 to 07-Apr-2016.
    File: https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z (Posts.xml)
    Date: 07-Apr-2024 11:06
    """

    def __init__(self, filters : list[Filter]= None):
        super().__init__('data/stackoverflow_post', filters=filters, checkpointPath='data/stackoverflow_post_parq.pkl')

    def preProcess(self) -> None:
        #This supplied version of Stack Overflow Post has already been preprocessed from the raw Posts.xml,
        #including labeling and filtering samples
        pass

    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample
    
    def getTag(self):
        return 'stackoverflow-post'
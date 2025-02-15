import pandas as pd
from tqdm import tqdm
from codetector.src.features.shared.data.models.mappable import MappableMixin
from codetector.src.features.shared.domain.entities.dataset.dataset import Dataset,DatasetBatch,Filter
from codetector.src.features.shared.domain.entities.samples.sample import Sample
from abc import abstractmethod

#XML
from xml.etree.ElementTree import Element, SubElement, fromstring, ParseError, indent, tostringlist
import re
import sys
from pathlib import Path

import numpy as np

class XMLDataset(Dataset):
    """
    Class handling importing and exporting XML datasets.
    Requires implementation of the pre-process function.
    """

    def __init__(self, folderPath:str, filters : list[Filter] = [],checkpointPath:str=None):
        super().__init__(filters, checkpointPath=checkpointPath)

        #Skip initialization if loaded from checkpoint
        if self.wasLoadedFromCheckpoint():
            return

        self.__folderPath = folderPath
        """
        The path to the folder that contains the data. '<folderPath>/Label_0.xml'
        """
        self.__count = -1
        """
        The number of samples contained in the dataset.
        """
        self.__readonly = False
        """
        If true, only read operations allowed.
        """
        self.__samples : list[Sample|MappableMixin]= []
        """
        The samples currently loaded in memory. This can either be samples waiting to be written to disk
        or samples that have been read from disk.
        """
        self.__SAVE_THRESHOLD = 5000
        """
        The threshold to save samples to disk.
        """
        self.__nextFile = 0
        """
        The index of the next label file that needs to be loaded.
        """

        self.__loadingIndex = 0
        """
        This is used to keep track of the items during loading.
        """


        assert issubclass(self.getContentType(),MappableMixin), f'Return type \'{self.getContentType().__name__}\' of {self.__class__.__name__}.getContentType() does not implement MappableMixin!'


    @abstractmethod
    def preProcess(self) -> None:
        """
        Called during XMLDataset.loadDataset().
        """
        pass


    @abstractmethod
    def getContentType(self) -> type[MappableMixin]:
        """
        Return the type of model implementation to use.
        """
        pass


    def __loadFile(self, fileNum:int) -> None:
        """
        Append the contents of the file to the __samples cache.
        Return `True` if file could be loaded.
        """


        #https://gist.github.com/lawlesst/4110923
        def invalid_xml_remove(c):
            #http://stackoverflow.com/questions/1707890/fast-way-to-filter-illegal-xml-unicode-chars-in-python
            illegal_unichrs = [ (0x00, 0x08), (0x0B, 0x1F), (0x7F, 0x84), (0x86, 0x9F),
                            (0xD800, 0xDFFF), (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF),
                            (0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), (0x3FFFE, 0x3FFFF),
                            (0x4FFFE, 0x4FFFF), (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                            (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), (0x9FFFE, 0x9FFFF),
                            (0xAFFFE, 0xAFFFF), (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                            (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), (0xFFFFE, 0xFFFFF),
                            (0x10FFFE, 0x10FFFF) ]

            illegal_ranges = ["%s-%s" % (chr(low), chr(high)) 
                        for (low, high) in illegal_unichrs 
                        if low < sys.maxunicode]

            illegal_xml_re = re.compile(u'[%s]' % u''.join(illegal_ranges))
            if illegal_xml_re.search(c) is not None:
                #Replace with space
                return ' '
            else:
                return c

        #https://gist.github.com/lawlesst/4110923
        def clean_char(char):
            """
            Function for remove invalid XML characters from
            incoming data.
            """
            #Get rid of the ctrl characters first.
            #http://stackoverflow.com/questions/1833873/python-regex-escape-characters
            char = re.sub('\x1b[^m]*m', '', char)
            #Clean up invalid xml
            char = invalid_xml_remove(char)
            replacements = [
                (u'\u201c', '\"'),
                (u'\u201d', '\"'),
                (u"\u001B", ' '), #http://www.fileformat.info/info/unicode/char/1b/index.htm
                (u"\u0019", ' '), #http://www.fileformat.info/info/unicode/char/19/index.htm
                (u"\u0016", ' '), #http://www.fileformat.info/info/unicode/char/16/index.htm
                (u"\u001C", ' '), #http://www.fileformat.info/info/unicode/char/1c/index.htm
                (u"\u0003", ' '), #http://www.utf8-chartable.de/unicode-utf8-table.pl?utf8=0x
                (u"\u000C", ' ')
            ]
            for rep, new_char in replacements:
                if char == rep:
                    #print ord(char), char.encode('ascii', 'ignore')
                    return new_char
            return char

        filePath = f'{self.__folderPath}/Label_{fileNum}.xml'

        if not Path(filePath).exists():
            return False

        with open(filePath, 'r') as file:
            for index, line in enumerate(file.readlines()):
                if index <= 1 or '</samples>' in line:
                    continue

                try:
                    elem = fromstring(line,)
                except ParseError:
                    elem = fromstring(''.join([clean_char(c) for c in line]))

                if elem.tag == "sample":
                    sample = self.getContentType().fromDict(elem.attrib)

######### POSSIBLE BUG with index range filter

                    if self.passesFilters(sample,index=self.__loadingIndex):
                        self.__samples.append(sample)
                        self.__loadingIndex += 1


        return True

    def loadDataset(self) -> None:
        self.preProcess()
        self.__readonly = True

        self.__loadFile(self.__nextFile)
        
        self.__nextFile += 1


    def loadBatch(self, size:int) -> DatasetBatch:
        if not self.__readonly:
            raise Exception('Trying to read from XML dataset in write mode!')

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

                done = not self.__loadFile(self.__nextFile) and len(self.__samples) == 0
                self.__nextFile += 1

            if not done and size > 0:
                samplesToReturn.extend(self.__samples[:size])
                self.__samples = self.__samples[size:]


        return DatasetBatch(samplesToReturn, done)

    def addSample(self, sample:Sample|MappableMixin) -> None:
        if self.__readonly:
            raise Exception('Trying to write to XML dataset in read-only mode!')
        
        self.__samples.append(sample)

        #Increment sample count
        self.__count = self.getCount() + 1

        if len(self.__samples) >= self.__SAVE_THRESHOLD:
            self.save()

    def save(self) -> None:
        if self.__readonly:
            raise Exception('Trying to save XML dataset in read-only mode!')
        
        if len(self.__samples) == 0:
            return

        root = Element("samples")
        for sample in self.__samples:
            SubElement(root,'sample',sample.toDict())
        indent(root, '  ')

        filePath = f'{self.__folderPath}/Label_{self.__nextFile}.xml'

        with open(filePath,'w+',encoding='utf8') as file:
            file.writelines(tostringlist(root,encoding="unicode",xml_declaration=True))

        self.__nextFile += 1

        #clear samples when saving
        self.__samples = []




    def __countSamples(self) -> int:
        """
        Count the samples in the xml file(s).
        """

        def _count_generator(reader):
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024 * 1024)

        fileNum = 0
        filePath = f'{self.__folderPath}/Label_{{}}.xml'

        lines = 0

        while Path(filePath.format(fileNum)).exists():
            with open(filePath.format(fileNum), 'rb') as file:
                c_generator = _count_generator(file.raw.read)
                lines += sum(buffer.count(b'\n') for buffer in c_generator) - 2
            fileNum += 1

        return lines

    def getCount(self) -> int:
        """
        Return the number of samples in the dataset.
        `WARNING: Filters are not applied to this implementation. Returns total count in dataset!`
        """
        
        if self.__count == -1:
            self.__count = self.__countSamples()

        return self.__count
    


    def toDataframe(self) -> pd.DataFrame:

        def __isInt(item:str) -> bool:
            try:
                a = int(item)
                return True
            except Exception:
                return False

        def __isFloat(item:str) -> bool:
            try:
                a = float(item)
                return True
            except Exception:
                return False



        bar = tqdm(total=self.getCount(), desc=f'Converting {self.__class__.__name__} to dataframe')

        stepSize = 100

        columns : dict[str,list] = {}
        column_types : dict[str,type] = {}

        batch = self.loadBatch(stepSize)
        while not batch.final or len(batch.samples) > 0:
            bar.update(len(batch.samples))
            for sample in batch.samples:
                asDict = sample.toDict()
                for key, value in asDict.items():

                    tempVal = value
                    if __isInt(tempVal):
                        tempVal = int(tempVal)

                        if not key in column_types:
                            column_types[key] = np.int16

                    elif __isFloat(tempVal):
                        tempVal = float(tempVal)

                        if not key in column_types:
                            column_types[key] = np.float32
                    else:
                        if not key in column_types and tempVal != 'None':
                            column_types[key] = str
                        elif tempVal == 'None':
                            tempVal = None

                    if not key in columns:
                        columns[key] = []
                    columns[key].append(tempVal)

            batch = self.loadBatch(stepSize)
            
        return pd.DataFrame(data=columns)
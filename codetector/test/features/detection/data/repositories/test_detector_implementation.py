from types import NoneType
import unittest
from unittest.mock import patch, MagicMock, call
from codetector.src.core.failure import DetectionFailure
from codetector.src.features.detection.data.repositories import DetectorRepositoryImplementation
from oslash import Right, Left

from codetector.src.features.detection.domain.entities.detector import BaseDetector
from codetector.src.features.detection.domain.entities.two_model_detector import TwoModelDetectorMixin
from codetector.src.features.shared.domain.entities.models.base_model import BaseModel
from codetector.src.features.shared.domain.entities.models.detector_model import DetectorMixin
from codetector.src.features.shared.domain.entities.samples import Sample, DetectionSample


class DetectorMock(BaseDetector):

    def initialize(self):
        raise NotImplementedError

    def detect(self, samples):
        raise NotImplementedError

    def setPrimaryModel(self, model):
        raise NotImplementedError

    def getTag(self) -> str:
        raise NotImplementedError


class TwoModelDetectorMock(BaseDetector,TwoModelDetectorMixin):

    def initialize(self):
        raise NotImplementedError

    def detect(self, samples):
        raise NotImplementedError

    def setPrimaryModel(self, model):
        raise NotImplementedError

    def getTag(self) -> str:
        raise NotImplementedError

    def setSecondaryModel(self, secondaryModel) -> None:
        raise NotImplementedError


class DetectorModelMock(BaseModel, DetectorMixin):
    def initialize(self, load:bool=False) -> None:
        raise NotImplementedError

    def getTag(self) -> str:
        raise NotImplementedError

    def getModelName(self) -> str:
        raise NotImplementedError

    def isLoaded(self) -> bool:
        raise NotImplementedError

    def load(self, outputHiddenStates=False) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def useSDP(self) -> bool:
        raise NotImplementedError

    def toPromptFormat(self, sample):
        raise NotImplementedError

    #Detector mixin

    def getLogits(self, sample, getHiddenStates=False):
        raise NotImplementedError

    def getLogitsBatch(self, samples, padding='do_not_pad'):
        raise NotImplementedError

    def getLoss(self, sample):
        raise NotImplementedError

    def getLossBatch(self, samples, padding=None): 
        raise NotImplementedError

    def getPadTokenId(self):
        raise NotImplementedError

    def isLargeModel(self):
        raise NotImplementedError


class DetectionSampleMock(DetectionSample):
    def __init__(self, content:str='content', value:float=0):
        super().__init__(content,
                         prompt='prompt',
                         originalPrompt='originalPrompt',
                         generatorTag='generatorTag',
                         datasetTag='datasetTag',
                         timestamp=0,
                         value=value,
                         detectorTag='detectorTag',
                         baseModelTag='baseModelTag',
                         sampleHash='sampleHash',
                         promptHash='promptHash',
                         topK=None, 
                         topP=0.95,
                         temperature=0.97,
                         secondaryModelTag=None,
                         maxLength=None)

class SampleMock(Sample):
    def __init__(self, content:str='content'):
        super().__init__(content,
                            'prompt',
                            'originalPrompt',
                            'generatorTag',
                            'datasetTag',
                            0,
                            None,
                            None,
                            None)
        
    def toDetectionSample(self, timestamp, value, detectorTag, baseModelTag, secondaryModelTag = None, maxLength = None):
        return DetectionSampleMock(content=self.content,value=value)

class TestDetectorImplementation(unittest.TestCase):

    def setUp(self):
        self.model : BaseModel|DetectorMixin = DetectorModelMock()
        self.datasource : BaseDetector = DetectorMock()
        self.repository : DetectorRepositoryImplementation = DetectorRepositoryImplementation()


    # addDetector
    # Should return DetectionFailure if non detector passed
    # Should return NoneResult if everything passed correctly

    def test_addDetectorNoDetector(self):
        """
        Should return DetectionFailure if incorrect type passed.
        """

        #Act
        result = self.repository.addDetector(None)

        #Assert
        self.assertIsInstance(result, Left)
        self.assertIsInstance(result._error, DetectionFailure)

    def test_addDetector(self):
        """
        Should return NoneResult.
        """

        #Act
        result = self.repository.addDetector(self.datasource)


        #Assert
        self.assertIsInstance(result, Right)
        self.assertIsInstance(result._value, NoneType)

    # registerPrimaryModel
    # Should return DetectionFailure if non detector passed
    # Should return NoneResult if everything passed correctly

    def test_registerPrimaryModelNoDetector(self):
        """
        Should return DetectionFailure if incorrect type passed.
        """

        #Act
        result = self.repository.registerPrimaryModel(None)

        #Assert
        self.assertIsInstance(result, Left)
        self.assertIsInstance(result._error, DetectionFailure)



    def test_registerPrimaryModel(self):
        """
        Should return NoneResult.
        """

        #Act
        result = self.repository.registerPrimaryModel(self.model)

        #Assert
        self.assertIsInstance(result, Right)
        self.assertIsInstance(result._value, NoneType)


    # registerSecondaryModels
    # Should return DetectionFailure if non detector passed
    # Should return NoneResult if everything passed correctly

    def test_registerSecondaryModelsNoDetectorPrimary(self):
        """
        Should return DetectionFailure if incorrect type passed.
        """

        #Act
        result = self.repository.registerSecondaryModels(None, [self.model])

        #Assert
        self.assertIsInstance(result, Left)
        self.assertIsInstance(result._error, DetectionFailure)


    def test_registerSecondaryModelsNoDetectorSecondary(self):
        """
        Should return DetectionFailure if incorrect type passed.
        """

        #Act
        result = self.repository.registerSecondaryModels(self.model, [None])

        #Assert
        self.assertIsInstance(result, Left)
        self.assertIsInstance(result._error, DetectionFailure)




    def test_registerSecondaryModels(self):
        """
        Should return NoneResult.
        """

        #Act
        result = self.repository.registerSecondaryModels(self.model,[self.model])

        #Assert
        self.assertIsInstance(result, Right)
        self.assertIsInstance(result._value, NoneType)


    # initialize
    # Should return DetectionFailure if detector raises exception.
    # Should return DetectionFailure if model raises exception.
    # Should return NoneResult and call initialize in added detectors and models.
    def test_initializeException(self):
        """
        Should return DetectionFailure if detector raises exception.
        """

        with patch.object(DetectorMock,'initialize') as mock:
            with patch.object(DetectorModelMock,'initialize') as mock2:
                #https://stackoverflow.com/questions/40736292/python-mock-raise-exception
                mock.side_effect = Exception()     
                mock2.return_value = None   
                
                #Act
                self.repository.addDetector(self.datasource)
                self.repository.registerPrimaryModel(self.model)
                self.repository.registerSecondaryModels(self.model,[self.model])
                result = self.repository.initialize()

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, DetectionFailure)
                mock.assert_called_once()
                mock2.assert_not_called()
    

    def test_initializeExceptionModel(self):
        """
        Should return DetectionFailure if model raises exception.
        """

        with patch.object(DetectorMock,'initialize') as mock:
            with patch.object(DetectorModelMock,'initialize') as mock2:
                #https://stackoverflow.com/questions/40736292/python-mock-raise-exception
                mock.return_value = None
                mock2.side_effect = Exception()
                
                #Act
                self.repository.addDetector(self.datasource)
                self.repository.registerPrimaryModel(self.model)
                self.repository.registerSecondaryModels(self.model,[self.model])
                result = self.repository.initialize()

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, DetectionFailure)
                mock.assert_called_once()
                mock2.assert_called_once()
    
    def test_initialize(self):
        """
        Should return NoneResult and call initialize in added generators.
        """

        returnVal = None

        with patch.object(DetectorMock,'initialize') as mock:
            with patch.object(DetectorModelMock,'initialize') as mock2:
                mock.return_value = returnVal        
                mock2.return_value = returnVal
                
                #Act
                self.repository.addDetector(self.datasource)
                self.repository.registerPrimaryModel(self.model)
                self.repository.registerSecondaryModels(self.model,[self.model])
                result = self.repository.initialize()

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock.assert_called_once()
                mock2.assert_called_once()

    # detect
    # Should return DetectionFailure if not initialized
    # Should return Failure if no detectors added
    # Should return Failure if no primary models added
    # Should return correct value and call [BaseDetector.setPrimaryModel] and [BaseDetector.detect] if single model detector
    # Shoulr return correct value and call [BaseDetector.setPrimaryModel],[TwoModelDetectorMixin.setSecondaryModel]
    # Should return None if a generator fails (Don't lose all progress)
    def test_detectNoInit(self):
        """
        Should return DetectionFailure if not initialized.
        """

        #Stub
        with patch.object(DetectorMock,'detect') as mock:
            mock.return_value = [None]
                
            #Act
            result = self.repository.detect([None])

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, DetectionFailure)
            mock.assert_not_called()

    def test_detectNoDetect(self):
        """
        Should return DetectionFailure if no detectors added.
        """
        #Stub
        with patch.object(DetectorMock,'detect') as mock:
            mock.return_value = [None]
                
            #Act
            self.repository.initialize()
            result = self.repository.detect([None])

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, DetectionFailure)
            mock.assert_not_called()

    def test_detectNoPrimary(self):
        """
        Should return DetectionFailure if no primary models added.
        """
        #Stub
        with patch.object(DetectorMock,'detect') as mock:
            mock.return_value = [None]
            with patch.object(DetectorMock,'initialize') as mock2:
                mock2.return_value = None
                #Act
                self.repository.addDetector(self.datasource)
                self.repository.initialize()
                result = self.repository.detect([None])

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, DetectionFailure)
                mock.assert_not_called()



    def test_detectSingleDetector(self):
        """
        Should return correct value and call [BaseDetector.setPrimaryModel] and [BaseDetector.detect] if single model detector.
        """

        inputValue = SampleMock()
        detectionValue = 4
        returnValue = inputValue.toDetectionSample(0,detectionValue,'ignored','ignored',)

        #Stub
        #There has to be a better way of doing this
        with patch.object(DetectorMock,'detect') as mock:
            mock.return_value = [detectionValue]
            with patch.object(DetectorMock,'setPrimaryModel') as mock2:
                mock2.return_value = None
                with patch.object(DetectorMock,'initialize') as mock3:
                    mock3.return_value = None
                    with patch.object(DetectorModelMock,'initialize') as mock4:
                        mock4.return_value = None
                        with patch.object(DetectorModelMock,'unload') as mock5:
                            mock5.return_value = None
                            with patch.object(DetectorModelMock,'getTag') as mock6:
                                mock6.return_value = 'baseModelTag'
                                with patch.object(DetectorMock,'getTag') as mock7:
                                    mock7.return_value = 'detectorTag'
                                    #Act
                                    self.repository.addDetector(self.datasource)
                                    self.repository.registerPrimaryModel(self.model)
                                    self.repository.initialize()
                                    result = self.repository.detect([inputValue])

                                    #Assert
                                    self.assertIsInstance(result, Right)
                                    self.assertListEqual(result._value, [returnValue])
                                    mock.assert_called_once_with([inputValue])
                                    mock2.assert_called_once_with(self.model)
                                    mock3.assert_called_once()
                                    mock4.assert_called_once()
                                    mock5.assert_called_once()
                                    mock6.assert_called_once()
                                    mock7.assert_called_once()

if __name__ == '__main__':
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.detection.domain.usecases.detect import DetectParameters, Detect
from codetector.src.core.failure import DetectionFailure

from oslash import Right,Left


class DetectionRepositoryMock(DetectorRepository):

    def initialize(self):
        raise NotImplementedError

    def detect(self, samples) :
        raise NotImplementedError

    def addDetector(self, detector):
        raise NotImplementedError

    def registerPrimaryModel(self, model):
        raise NotImplementedError

    def registerSecondaryModels(self, model, secondaryModels):
        raise NotImplementedError

    def setMaxDetectionLength(self, length):
        raise NotImplementedError


class TestDetect(unittest.TestCase):

    def setUp(self):
        self.repository : DetectorRepository = DetectionRepositoryMock()
        self.usecase : Detect = Detect(self.repository)

    @patch.object(DetectionRepositoryMock, 'setMaxDetectionLength')
    @patch.object(DetectionRepositoryMock, 'detect')
    def test_callRepositoryFunction(self, mock_method:MagicMock, mock_method2:MagicMock):
        """
        Should call [DetectorRepository.detect] and [DetectorRepository.setMaxDetectionLength].
        """

        params = DetectParameters([None],maxLength=5)


        #Stub
        mock_method.return_value = Right([None])
        mock_method2.return_value = Right(None)

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, Right([None]))
        mock_method.assert_called_once_with(params.samples)
        mock_method2.assert_called_once_with(params.maxLength)


    @patch.object(DetectionRepositoryMock, 'setMaxDetectionLength')
    @patch.object(DetectionRepositoryMock, 'detect')
    def test_returnFailure(self, mock_method:MagicMock, mock_method2:MagicMock):
        """
        Should return Failure if register primary model fails.
        """

        params = DetectParameters([None],maxLength=5)
        return_value = Left(DetectionFailure('Some error',-1))

        #Stub
        mock_method.return_value = Right([None])
        mock_method2.return_value = return_value

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, return_value)
        mock_method.assert_not_called()
        mock_method2.assert_called_once_with(params.maxLength)



if __name__ == '__main__':
    unittest.main()
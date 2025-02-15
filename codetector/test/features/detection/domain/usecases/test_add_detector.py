import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.detection.domain.usecases.add_detector import AddDetectorParameters, AddDetector
from oslash import Right


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


class TestAddDetector(unittest.TestCase):

    def setUp(self):
        self.repository : DetectorRepository = DetectionRepositoryMock()
        self.usecase : AddDetector = AddDetector(self.repository)

    @patch.object(DetectionRepositoryMock, 'addDetector')
    def test_callRepositoryFunction(self, mock_method:MagicMock):
        """
        Should call [DetectorRepository.addDetector] once.
        """

        params = AddDetectorParameters(None)


        #Stub
        mock_method.return_value = Right(None)

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, Right(None))
        mock_method.assert_called_once_with(params.detector)


if __name__ == '__main__':
    unittest.main()
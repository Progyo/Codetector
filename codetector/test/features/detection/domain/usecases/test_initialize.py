import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.detection.domain.usecases.initialize import Initialize
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

class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.repository : DetectorRepository = DetectionRepositoryMock()
        self.usecase : Initialize = Initialize(self.repository)

    @patch.object(DetectionRepositoryMock, 'initialize')
    def test_callRepositoryFunction(self, mock_method:MagicMock):
        """
        Should call [DetectorRepository.initialize] once.
        """

        #Stub
        mock_method.return_value = Right(None)

        #Act
        result = self.usecase()

        #Assert
        self.assertEqual(result, Right(None))
        mock_method.assert_called_once()


if __name__ == '__main__':
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.detection.domain.usecases.set_max_length import SetMaxLengthParameters, SetMaxLength
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


class TestSetMaxLength(unittest.TestCase):

    def setUp(self):
        self.repository : DetectorRepository = DetectionRepositoryMock()
        self.usecase : SetMaxLength = SetMaxLength(self.repository)

    @patch.object(DetectionRepositoryMock, 'setMaxDetectionLength')
    def test_callRepositoryFunction(self, mock_method:MagicMock):
        """
        Should call [DetectorRepository.setMaxDetectionLength] once.
        """

        params = SetMaxLengthParameters(5)


        #Stub
        mock_method.return_value = Right(None)

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, Right(None))
        mock_method.assert_called_once_with(params.length)


if __name__ == '__main__':
    unittest.main()
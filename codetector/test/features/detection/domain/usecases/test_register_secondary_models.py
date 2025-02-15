import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.detection.domain.usecases.register_secondary_models import RegisterSecondaryModelsParameters, RegisterSecondaryModels
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


class TestRegisterSecondaryModels(unittest.TestCase):

    def setUp(self):
        self.repository : DetectorRepository = DetectionRepositoryMock()
        self.usecase : RegisterSecondaryModels = RegisterSecondaryModels(self.repository)

    @patch.object(DetectionRepositoryMock, 'registerSecondaryModels')
    def test_callRepositoryFunction(self, mock_method:MagicMock):
        """
        Should call [DetectorRepository.registerSecondaryModel] for each model.
        """

        params = RegisterSecondaryModelsParameters({None:[None]})


        #Stub
        mock_method.return_value = Right(None)

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, Right(None))
        mock_method.assert_called_once_with(None,[None])



    @patch.object(DetectionRepositoryMock, 'registerSecondaryModels')
    def test_returnFailure(self, mock_method:MagicMock):
        """
        Should return Failure if register primary model fails.
        """

        params = RegisterSecondaryModelsParameters({None:[None]})
        return_value = Left(DetectionFailure('Some error',-1))

        #Stub
        mock_method.return_value = return_value

        #Act
        result = self.usecase(params)

        #Assert
        self.assertEqual(result, return_value)
        mock_method.assert_called_once_with(None,[None])



if __name__ == '__main__':
    unittest.main()
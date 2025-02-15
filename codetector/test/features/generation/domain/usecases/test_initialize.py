import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.generation.domain.repositories import GeneratorRepository
from codetector.src.features.generation.domain.usecases.initialize import Initialize
# from codetector.src.core import Result, NoneResult
from oslash import Right


class GeneratorRepositoryMock(GeneratorRepository):
    def initialize(self):
        raise NotImplementedError

    def generateSingle(self, sample):
        raise NotImplementedError

    def generateBatch(self, samples):
        raise NotImplementedError

    def setTemperature(self, temperature):
        raise NotImplementedError

    def setTopK(self, k):
        raise NotImplementedError

    def setTopP(self, p):
        raise NotImplementedError

    def setMaxOutputLength(self, length):
        raise NotImplementedError

    def setGenerateCount(self, count):
        raise NotImplementedError

    def addGenerator(self, generator):
        raise NotImplementedError

    def setBestOf(self, bestOf, heuristic):
        raise NotImplementedError

    def enableDynamicSize(self, enable):
        raise NotImplementedError


class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.repository : GeneratorRepository = GeneratorRepositoryMock()
        self.usecase : Initialize = Initialize(self.repository)

    #@patch(__name__+'.GeneratorRepositoryMock.initialize')
    @patch.object(GeneratorRepositoryMock, 'initialize')
    def test_callRepositoryFunction(self, mock_method:MagicMock):
        """
        Should call [GeneratorRepository.initialize] once.
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
import unittest
from unittest.mock import patch, MagicMock
from codetector.src.features.generation.domain.repositories import GeneratorRepository
from codetector.src.features.generation.domain.usecases.generate import GenerateParameters, Generate
from oslash import Right, Left


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


class TestGenerate(unittest.TestCase):

    def setUp(self):
        self.repository : GeneratorRepository = GeneratorRepositoryMock()
        self.usecase : Generate = Generate(self.repository)


    def test_callGenerateSingle(self):
        """
        Should only call [GeneratorRepository.generateSingle] once if batch=False.
        """

        params = GenerateParameters([None],False)
        returnVal = Right([None])

        #Stub
        with patch.object(GeneratorRepositoryMock,'generateSingle') as mock:
            mock.return_value = returnVal
            with patch.object(GeneratorRepositoryMock,'generateBatch') as mock2:
                
                #Act
                result = self.usecase(params)

                #Assert
                self.assertEqual(result, returnVal)
                mock.assert_called_once_with(params.samples[0])
                mock2.assert_not_called()


    def test_callGenerateBatch(self):
        """
        Should call [GeneratorRepository.generateBatch] once if batch=True.
        """

        params = GenerateParameters([None],True)
        returnVal = Right([[None]])

        #Stub
        with patch.object(GeneratorRepositoryMock,'generateBatch') as mock:
            mock.return_value = returnVal
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                
                #Act
                result = self.usecase(params)

                #Assert
                self.assertEqual(result, returnVal)
                mock.assert_called_once_with(params.samples)
                mock2.assert_not_called()


    def test_callSetTemperature(self):
        """
        Should call [GeneratorRepository.setTemperature] once if temperature!=None.
        """

        params = GenerateParameters([None],False,temperature=0.5)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTemperature') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.temperature)
                mock2.assert_called_once()

    def test_callSetTemperatureFailed(self):
        """
        Should return left failure if [GeneratorRepository.setTemperature] fails.
        """

        params = GenerateParameters([None],False,temperature=0.5)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTemperature') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.temperature)

    def test_callSetTopK(self):
        """
        Should call [GeneratorRepository.setTopK] once if topK!=None.
        """

        params = GenerateParameters([None],False,topK=40)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTopK') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.topK)
                mock2.assert_called_once()

    def test_callSetTopKFailed(self):
        """
        Should return left failure if [GeneratorRepository.setTopK] fails.
        """

        params = GenerateParameters([None],False,topK=40)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTopK') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.topK)


    def test_callSetTopP(self):
        """
        Should call [GeneratorRepository.setTopP] once if topP!=None.
        """

        params = GenerateParameters([None],False,topP=0.5)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTopP') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.topP)
                mock2.assert_called_once()

    def test_callSetTopPFailed(self):
        """
        Should return left failure if [GeneratorRepository.setTopP] fails.
        """

        params = GenerateParameters([None],False,topP=0.5)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setTopP') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.topP)


    def test_callSetGenerateCount(self):
        """
        Should call [GeneratorRepository.setGenerateCount] once if generateCount!=None.
        """

        params = GenerateParameters([None],False,generateCount=1)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setGenerateCount') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.generateCount)
                mock2.assert_called_once()


    def test_callSetGenerateCountFailed(self):
        """
        Should return left failure if [GeneratorRepository.setGenerateCount] fails.
        """

        params = GenerateParameters([None],False,generateCount=1)

        #Stub
        with patch.object(GeneratorRepositoryMock,'setGenerateCount') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.generateCount)


    def test_callSetBestOf(self):
        """
        Should call [GeneratorRepository.setBestOf] once if bestOf!=None.
        """

        params = GenerateParameters([None],False,bestOf=(1,None))

        #Stub
        with patch.object(GeneratorRepositoryMock,'setBestOf') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.bestOf[0],heuristic=params.bestOf[1])
                mock2.assert_called_once()


    def test_callSetBestOfFailed(self):
        """
        Should return left failure if [GeneratorRepository.setBestOf] fails.
        """

        params = GenerateParameters([None],False,bestOf=(1,None))

        #Stub
        with patch.object(GeneratorRepositoryMock,'setBestOf') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.bestOf[0],heuristic=params.bestOf[1])


    def test_callEnableDynamicSize(self):
        """
        Should call [GeneratorRepository.enableDynamicSize] once if dynamicSize!=None.
        """

        params = GenerateParameters([None],False,dynamicSize=True)

        #Stub
        with patch.object(GeneratorRepositoryMock,'enableDynamicSize') as mock:
            mock.return_value = Right(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Right)
                mock.assert_called_once_with(params.dynamicSize)
                mock2.assert_called_once()

    def test_callEnableDynamicSizeFailed(self):
        """
        Should return left failure if [GeneratorRepository.enableDynamicSize] fails.
        """

        params = GenerateParameters([None],False,dynamicSize=True)

        #Stub
        with patch.object(GeneratorRepositoryMock,'enableDynamicSize') as mock:
            mock.return_value = Left(None)
            with patch.object(GeneratorRepositoryMock,'generateSingle') as mock2:
                mock2.return_value = Right([None])

                #Act
                result = self.usecase(params)

                #Assert
                self.assertIsInstance(result, Left)
                mock.assert_called_once_with(params.dynamicSize)


    def test_dontCallOptional(self):
        """
        Should not call [GeneratorRepository.setTopK, GeneratorRepository.setTopP, GeneratorRepository.setTemperature,
        GeneratorRepository.setGenerateCount, GeneratorRepository.setBestOf, GeneratorRepository.enableDynamicSize]
        if topK==None, topP==None, temperature==None, generateCount==None, bestOf==None, dynamicSize==None.
        """

        params = GenerateParameters([None],False)

        #Stub
        with patch.object(GeneratorRepositoryMock,'generateSingle') as mock:
            mock.return_value = Right([None])
            
            with patch.object(GeneratorRepositoryMock,'setTopK') as mock2:
                 with patch.object(GeneratorRepositoryMock,'setTopP') as mock3:
                      with patch.object(GeneratorRepositoryMock,'setTemperature') as mock4:
                          with patch.object(GeneratorRepositoryMock,'setGenerateCount') as mock5:
                              with patch.object(GeneratorRepositoryMock,'setBestOf') as mock6:
                                  with patch.object(GeneratorRepositoryMock,'enableDynamicSize') as mock7:
                

                                    #Act
                                    result = self.usecase(params)

                                    #Assert
                                    self.assertIsInstance(result, Right)
                                    mock2.assert_not_called()
                                    mock3.assert_not_called()
                                    mock4.assert_not_called()
                                    mock5.assert_not_called()
                                    mock6.assert_not_called()
                                    mock7.assert_not_called()
                                    mock.assert_called_once()

if __name__ == '__main__':
    unittest.main()
from types import NoneType
import unittest
from unittest.mock import patch, MagicMock, call
from codetector.src.core.failure import GenerateFailure
from codetector.src.features.generation.data.repositories import GeneratorRepositoryImplementation
from oslash import Right, Left

from codetector.src.features.shared.domain.entities.models.base_model import BaseModel
from codetector.src.features.shared.domain.entities.models.generator_model import GeneratorMixin
from codetector.src.features.shared.domain.entities.samples.sample import Sample

class GeneratorModelMock(BaseModel, GeneratorMixin):
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

    #Generator mixin

    def generateSingle(self, sample) -> list:
        raise NotImplementedError

    def generateBatch(self, sampleslist) -> list[list]:
        raise NotImplementedError

    def setTemperature(self, temperature) -> None:
        raise NotImplementedError

    def setTopK(self, k) -> None:
        raise NotImplementedError

    def setTopP(self, p:float) -> None:
        raise NotImplementedError

    def setMaxOutputLength(self, length) -> None:
        raise NotImplementedError

    def setGenerateCount(self, count) -> None:
        raise NotImplementedError

    def setBestOf(self, bestOf, heuristic) -> None:
        raise NotImplementedError

    def enableDynamicSize(self, enable) -> None:
        raise NotImplementedError

    def supportsSample(self, sample) -> bool:
        raise NotImplementedError


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
        raise NotImplementedError

class TestGeneratorImplementation(unittest.TestCase):

    def setUp(self):
        self.datasource : GeneratorMixin = GeneratorModelMock()
        self.repository : GeneratorRepositoryImplementation = GeneratorRepositoryImplementation()


    # addGenerator
    # Should return GenerateFailure if non generator passed
    # Should return NoneResult if everything passed correctly

    def test_addGeneratorNoGenerator(self):
        """
        Should return GenerateFailure if incorrect type passed.
        """

        #Act
        result = self.repository.addGenerator(None)

        #Assert
        self.assertIsInstance(result, Left)
        self.assertIsInstance(result._error, GenerateFailure)

    def test_addGenerator(self):
        """
        Should return NoneResult.
        """

        #Act
        result = self.repository.addGenerator(self.datasource)

        #Assert
        self.assertIsInstance(result, Right)
        self.assertIsInstance(result._value, NoneType)

    # initialize
    # Should return GenerateFailure if generator raises exception.
    # Should return NoneResult and call initialize in added generators.
    def test_initializeException(self):
        """
        Should return GenerateFailure if generator raises exception.
        """

        with patch.object(GeneratorModelMock,'initialize') as mock:

            #https://stackoverflow.com/questions/40736292/python-mock-raise-exception
            mock.side_effect = Exception()        
            
            #Act
            self.repository.addGenerator(self.datasource)
            result = self.repository.initialize()

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, GenerateFailure)
            mock.assert_called_once()
    
    
    def test_initialize(self):
        """
        Should return NoneResult and call initialize in added generators.
        """

        returnVal = None

        with patch.object(GeneratorModelMock,'initialize') as mock:

            mock.return_value = returnVal        
            
            #Act
            self.repository.addGenerator(self.datasource)
            result = self.repository.initialize()

            #Assert
            self.assertIsInstance(result, Right)
            self.assertIsInstance(result._value, NoneType)
            mock.assert_called_once()

    # generateSingle
    # Should return GenerateFailure if not initialized
    # Should return Failure if no generators added
    # Should return correct value if generator is stubbed
    # Should skip sample if supportsSample returns False
    # Should return None if a generator fails (Don't lose all progress)
    def test_generateSingleNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        returnVal = GenerateFailure('',-1)

        #Stub
        with patch.object(GeneratorModelMock,'generateSingle') as mock:
            mock.return_value = returnVal
                
            #Act
            result = self.repository.generateSingle(None)

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, GenerateFailure)
            mock.assert_not_called()

    def test_generateSingleNoGen(self):
        """
        Should return GenerateFailure if no generators added.
        """

        returnVal = GenerateFailure('',-1)

        #Stub
        with patch.object(GeneratorModelMock,'generateSingle') as mock:
            mock.return_value = returnVal
                
            #Act
            self.repository.initialize()
            result = self.repository.generateSingle(None)

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, GenerateFailure)
            mock.assert_not_called()


    def test_generateSingleSkipSample(self):
        """
        Should skip sample if supportsSample returns False.
        """

        returnVal = [SampleMock()]
        sample = SampleMock()

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'useSDP') as mock2:
                mock2.return_value = False
                with patch.object(GeneratorModelMock,'supportsSample') as mock3:
                    mock3.return_value = False
                    with patch.object(GeneratorModelMock,'generateSingle') as mock4:
                        mock4.return_value = returnVal
                        with patch.object(GeneratorModelMock,'unload') as mock5:
                            mock5.return_value = None
                            #Act
                            self.repository.addGenerator(self.datasource)
                            self.repository.initialize()
                            result = self.repository.generateSingle(sample)

                            #Assert
                            self.assertIsInstance(result, Right)
                            self.assertTrue(len(result._value)==1)
                            self.assertIsInstance(result._value[0], NoneType)
                            mock.assert_called_once()
                            mock3.assert_called_once_with(sample)
                            mock4.assert_not_called()
                            mock5.assert_called_once()


    def test_generateSingleCorrectVal(self):
        """
        Should return correct value if generator is stubbed.
        """

        returnVal = [SampleMock()]
        sample = SampleMock()

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'useSDP') as mock2:
                mock2.return_value = False
                with patch.object(GeneratorModelMock,'supportsSample') as mock3:
                    mock3.return_value = True
                    with patch.object(GeneratorModelMock,'generateSingle') as mock4:
                        mock4.return_value = returnVal
                        with patch.object(GeneratorModelMock,'unload') as mock5:
                            mock5.return_value = None
                            
                            #Act
                            self.repository.addGenerator(self.datasource)
                            self.repository.initialize()
                            result = self.repository.generateSingle(sample)

                            #Assert
                            self.assertIsInstance(result, Right)
                            self.assertTrue(issubclass(result._value[0][0].__class__, Sample))
                            mock.assert_called_once()
                            mock3.assert_called_once_with(sample)
                            mock4.assert_called_once_with(sample)
                            mock5.assert_called_once()




    # setTemperature
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setTemperatureNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        temp = 0.97
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTemperature') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setTemperature(temp)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setTemperatureExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        temp = 0.97
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTemperature') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTemperature(temp)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(temp)

    def test_setTemperature(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        temp = 0.97
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTemperature') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTemperature(temp)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(temp)


    # setTopK
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setTopKNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        topK = 40
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopK') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setTopK(topK)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setTopKExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        topK = 40
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopK') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTopK(topK)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(topK)

    def test_setTopK(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        topK = 40
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopK') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTopK(topK)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(topK)


    # setTopP
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setTopPNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        topP = 0.95
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopP') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setTopP(topP)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setTopPExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        topP = 0.95
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopP') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTopP(topP)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(topP)

    def test_setTopP(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        topP = 0.95
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setTopP') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setTopP(topP)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(topP)


    # setMaxOutputLength
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setMaxOutputLengthNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        maxLen = 1024
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setMaxOutputLength') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setMaxOutputLength(maxLen)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setMaxOutputLengthExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        maxLen = 1024
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setMaxOutputLength') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setMaxOutputLength(maxLen)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(maxLen)

    def test_setMaxOutputLength(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        maxLen = 1024
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setMaxOutputLength') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setMaxOutputLength(maxLen)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(maxLen)



    # setGenerateCount
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setGenerateCountNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        generateCount = 1
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setGenerateCount') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setGenerateCount(generateCount)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setGenerateCountExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        generateCount = 1
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setGenerateCount') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setGenerateCount(generateCount)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(generateCount)

    def test_setGenerateCount(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        generateCount = 1
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setGenerateCount') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setGenerateCount(generateCount)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(generateCount)


    # setBestOf
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_setBestOfNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        bestOf = 1
        heuristic = lambda x : x
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setBestOf') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.setBestOf(bestOf,heuristic=heuristic)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_setBestOfExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        bestOf = 1
        heuristic = lambda x : x
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setBestOf') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setBestOf(bestOf,heuristic=heuristic)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(bestOf,heuristic)

    def test_setBestOfCount(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        bestOf = 1
        heuristic = lambda x : x
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'setBestOf') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.setBestOf(bestOf,heuristic=heuristic)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(bestOf,heuristic)


    # enableDynamicSize
    # Should return GenerateFailure if not initialized
    # Should return GenerateFailure if exception in one of the generators
    # Should return NoneResult if generator is successful in all
    def test_enableDynamicSizeNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        enable = True
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'enableDynamicSize') as mock2:
                mock2.return_value = returnVal

                #Act
                result = self.repository.enableDynamicSize(enable)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_not_called()

    def test_enableDynamicSizeExcep(self):
        """
        Should return GenerateFailure if exception in one of the generators.
        """

        enable = True
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'enableDynamicSize') as mock2:
                mock2.side_effect = Exception()

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.enableDynamicSize(enable)

                #Assert
                self.assertIsInstance(result, Left)
                self.assertIsInstance(result._error, GenerateFailure)
                mock2.assert_called_once_with(enable)

    def test_enableDynamicSizeCount(self):
        """
        Should return NoneResult if generator is successful in all.
        """

        enable = True
        returnVal = None

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'enableDynamicSize') as mock2:
                mock2.return_value = returnVal

                #Act
                self.repository.addGenerator(self.datasource)
                self.repository.initialize()
                result = self.repository.enableDynamicSize(enable)

                #Assert
                self.assertIsInstance(result, Right)
                self.assertIsInstance(result._value, NoneType)
                mock2.assert_called_once_with(enable)

    # groupBySize
    def test_groupBySizeInt(self):
        """
        Should return correct groups when using int.
        """

        #https://stackoverflow.com/questions/15453283/testing-private-methods-in-python-unit-test-or-functional-test


        inputSamples : list[Sample] = [SampleMock('content '*1024),
                                       SampleMock('content'),
                                       SampleMock('content '*512),
                                       SampleMock('content '*500),
                                       SampleMock('content '*1000),
                                       SampleMock('content '*128),
                                       SampleMock('content '*10)]
        
        maxDif = 100
        expectedOutputSamples : list[list[Sample]] = [[inputSamples[1],inputSamples[6]],
                                               [inputSamples[5]],
                                               [inputSamples[3],inputSamples[2]],
                                               [inputSamples[4],inputSamples[0]]]

        expectedOutputIndicies : list[list[int]] = [[1,6],
                                                    [5],
                                                    [3,2],
                                                    [4,0]]



        #Stub
        #Act
        result : tuple[list[list[Sample]],list[list[int]]] = self.repository._GeneratorRepositoryImplementation__groupBySize(inputSamples, maxDif)

        #Assert
        self.assertListEqual(expectedOutputSamples, result[0])
        self.assertListEqual(expectedOutputIndicies, result[1])


    def test_groupBySizeFloat(self):
        """
        Should return correct groups when using float.
        """

        #https://stackoverflow.com/questions/15453283/testing-private-methods-in-python-unit-test-or-functional-test


        inputSamples : list[Sample] = [SampleMock('content '*1024),
                                       SampleMock('content'),
                                       SampleMock('content '*512),
                                       SampleMock('content '*500),
                                       SampleMock('content '*1000),
                                       SampleMock('content '*128),
                                       SampleMock('content '*10)]
        
        maxDif = 0.25
        expectedOutputSamples : list[list[Sample]] = [[inputSamples[1]],
                                                    [inputSamples[6]],
                                                    [inputSamples[5]],
                                                    [inputSamples[3],inputSamples[2]],
                                                    [inputSamples[4],inputSamples[0]]]

        expectedOutputIndicies : list[list[int]] = [[1],
                                                    [6],
                                                    [5],
                                                    [3,2],
                                                    [4,0]]



        #Stub
        #Act
        result : tuple[list[list[Sample]],list[list[int]]] = self.repository._GeneratorRepositoryImplementation__groupBySize(inputSamples, maxDif)

        #Assert
        self.assertListEqual(expectedOutputSamples, result[0])
        self.assertListEqual(expectedOutputIndicies, result[1])


    # generateBatch
    # Should return GenerateFailure if not initialized
    # Should return Failure if no generators added
    # Should return correct value if generator is stubbed
    # Should return list of None if not supported

    # v Individual generator should be responsible for returning None as well
    # Should return None if a generator fails (Don't lose all progress)
    def test_generateBatchNoInit(self):
        """
        Should return GenerateFailure if not initialized.
        """

        returnVal = GenerateFailure('',-1)

        #Stub
        with patch.object(GeneratorModelMock,'generateBatch') as mock:
            mock.return_value = returnVal
                
            #Act
            result = self.repository.generateBatch([])

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, GenerateFailure)
            mock.assert_not_called()

    def test_generateBatchNoGen(self):
        """
        Should return GenerateFailure if no generators added.
        """

        returnVal = GenerateFailure('',-1)

        #Stub
        with patch.object(GeneratorModelMock,'generateBatch') as mock:
            mock.return_value = returnVal
                
            #Act
            self.repository.initialize()
            result = self.repository.generateBatch([])

            #Assert
            self.assertIsInstance(result, Left)
            self.assertIsInstance(result._error, GenerateFailure)
            mock.assert_not_called()


    def test_generateBatchCorrectVal(self):
        """
        Should return correct value if generator is stubbed.
        """


        inputSamples : list[Sample] = [SampleMock('content '*1024),
                                       SampleMock('content'),
                                       SampleMock('content '*512),
                                       SampleMock('content '*500),
                                       SampleMock('content '*1000),
                                       SampleMock('content '*128),
                                       SampleMock('content '*10)]

        # After grouping using maxDiff 0.25
        expectedCalls : list[list[Sample]] = [[inputSamples[1]],
                                            [inputSamples[6]],
                                            [inputSamples[5]],
                                            [inputSamples[3],inputSamples[2]],
                                            [inputSamples[4],inputSamples[0]]]

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'useSDP') as mock2:
                mock2.return_value = False
                with patch.object(GeneratorModelMock,'supportsSample') as mock3:
                    mock3.return_value = True
                    with patch.object(GeneratorModelMock,'generateBatch') as mock4:
                        # mock3.return_value = returnVal
                        mock4.side_effect = lambda x: list(map(lambda y: [y],x)) #lambda x: [[SampleMock()]]*len(x)
                        with patch.object(GeneratorModelMock,'unload') as mock5:
                            mock5.return_value = None    
                            #Act
                            self.repository.addGenerator(self.datasource)
                            self.repository.initialize()
                            result = self.repository.generateBatch(inputSamples)

                            #Assert

                            #Correct return type
                            self.assertIsInstance(result, Right)
                            self.assertTrue(issubclass(result._value[0][0][0].__class__, Sample))
                            
                            #Correct size
                            self.assertTrue(len(result._value)==1)
                            self.assertTrue(len(result._value[0])==len(inputSamples))
                            
                            #Correct generateBatch calls
                            mock4.assert_has_calls(list(map(lambda x: call(x), [expectedCalls[i] for i in range(0,3)])))
                            self.assertListEqual(result._value[0], list(map(lambda x: [x], inputSamples)))
                            
                            #Correct support sample calls
                            mock3.assert_has_calls(list(map(lambda x: call(x), [inputSamples[i] for i in range(0,3)])), any_order=True)

                            mock5.assert_called_once()


    def test_generateBatchReturnNoneList(self):
        """
        Should return list of None if not supported.
        """


        inputSamples : list[Sample] = [SampleMock('content '*1024),
                                       SampleMock('content'),
                                       SampleMock('content '*512),
                                       SampleMock('content '*500),
                                       SampleMock('content '*1000),
                                       SampleMock('content '*128),
                                       SampleMock('content '*10)]

        # After grouping using maxDiff 0.25
        expectedCalls : list[list[Sample]] = [[inputSamples[1]],
                                            [inputSamples[6]],
                                            [inputSamples[5]],
                                            [inputSamples[3],inputSamples[2]],
                                            [inputSamples[4],inputSamples[0]]]

        #Stub
        with patch.object(GeneratorModelMock,'initialize') as mock:
            mock.return_value = None
            with patch.object(GeneratorModelMock,'useSDP') as mock2:
                mock2.return_value = False
                with patch.object(GeneratorModelMock,'supportsSample') as mock3:
                    mock3.return_value = False
                    with patch.object(GeneratorModelMock,'generateBatch') as mock4:
                        # mock3.return_value = returnVal
                        mock4.side_effect = lambda x: list(map(lambda y: [y],x)) #lambda x: [[SampleMock()]]*len(x)
                        with patch.object(GeneratorModelMock,'unload') as mock5:
                            mock5.return_value = None    
                            #Act
                            self.repository.addGenerator(self.datasource)
                            self.repository.initialize()
                            result = self.repository.generateBatch(inputSamples)

                            #Assert

                            #Correct return type
                            self.assertIsInstance(result, Right)
                            self.assertTrue(issubclass(result._value[0][0][0].__class__, NoneType))
                            
                            #Correct size
                            self.assertTrue(len(result._value)==1)
                            self.assertTrue(len(result._value[0])==len(inputSamples))
                            
                            #Correct generateBatch calls
                            mock4.assert_has_calls(list(map(lambda x: call(x), [[None]*len(expectedCalls[i]) for i in range(0,3)])))
                            self.assertListEqual(result._value[0], [[None]]*len(inputSamples))
                            
                            #Correct support sample calls
                            mock3.assert_has_calls(list(map(lambda x: call(x), [inputSamples[i] for i in range(0,3)])), any_order=True)


                            mock5.assert_called_once()

if __name__ == '__main__':
    unittest.main()
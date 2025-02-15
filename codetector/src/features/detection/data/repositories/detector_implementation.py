from codetector.src.core.failure import DetectionFailure
from codetector.src.core.typedef import NoneResult, Result
from codetector.src.features.detection.domain.entities.detector import BaseDetector
from codetector.src.features.detection.domain.entities.two_model_detector import TwoModelDetectorMixin
from codetector.src.features.shared.data.models.tokenizers.tiktoken import TikTokenTokenizer
from codetector.src.features.detection.domain.repositories import DetectorRepository
from codetector.src.features.shared.domain.entities import Sample, DetectionSample, BaseModel, DetectorMixin
from codetector.src.features.shared.domain.entities.tokenizer import Tokenizer

from oslash import Right, Left
import time

try:
    import torch as torch
except ModuleNotFoundError:
    torch = None

class DetectorRepositoryImplementation(DetectorRepository):
    """
    Implementation of the detector repository.
    """

    def __init__(self):
        super().__init__()

        self.__detectors : list[BaseDetector|TwoModelDetectorMixin] = []
        """
        The detectors being used and managed.
        """

        self.__primaryModels : list[BaseModel|DetectorMixin] = []
        """
        The primary models used by the detectors.
        """

        self.__secondaryModels : dict[BaseModel|DetectorMixin,list[BaseModel|DetectorMixin]] = {}
        """
        The secondary model combinations that two model detectors can use.
        """

        self.__initialized : bool = False


        self.__maxDetectionLength : int = None
        """
        The maximum length of sample that is considered for detection.
        """


    def initialize(self) -> NoneResult:
        """
        Initialize the detector repository.
        Also initializes all detectors and registered models.
        """ 
        if not self.__initialized:

            for detector in self.__detectors:
                try:
                    detector.initialize()
                except Exception as e:
                    return Left(DetectionFailure(f'{detector.__class__.__name__} had an exception "{e}" during initialization',-1))

            for model in self.__primaryModels:
                try:
                    model.initialize()
                except Exception as e:
                    return Left(DetectionFailure(f'{model.__class__.__name__} had an exception "{e}" during initialization',-1))


            self.__initialized = True

        return Right(None)

    def detect(self, samples:list[Sample]) -> Result[list[DetectionSample]]:
        """
        Analyze the samples using the loaded detectors and registered base models and secondary models.
        """
        if not self.__initialized:
            return Left(DetectionFailure('Detector not initialized',-1))
        
        if len(self.__detectors) == 0:
            return Left(DetectionFailure('No detectors added',-1))

        if len(self.__primaryModels) == 0:
            return Left(DetectionFailure('No primary models added',-1))


        toReturn : list[DetectionSample] = []

        #Loop through primary models in the outerloop to prevent
        #Constant model loading and unloading
        for primaryModel in self.__primaryModels:
            for detector in self.__detectors:

                detector.setPrimaryModel(primaryModel)

                if issubclass(detector.__class__,TwoModelDetectorMixin) and primaryModel in self.__secondaryModels:
                    for secondaryModel in self.__secondaryModels[primaryModel]:
                        detector.setSecondaryModel(secondaryModel)
                        try:
                            values = detector.detect(samples)
                            toReturn += list(map(lambda x:x[0].toDetectionSample(timestamp=int(time.time()),
                                                                                value=x[1],
                                                                                detectorTag=detector.getTag(),
                                                                                baseModelTag=primaryModel.getTag(),
                                                                                secondaryModelTag=secondaryModel.getTag(),
                                                                                maxLength=self.__maxDetectionLength), zip(samples,values)))
                        except Exception as e:
                            return DetectionFailure(f'{detector.getTag()} had an exception "{e}" during detection with models: {primaryModel.getTag()} and {secondaryModel.getTag()}',-1)
                        #May remove this later
                        secondaryModel.unload()
                else:
                    try:
                        values = detector.detect(samples)
                        zipped = list(zip(samples,values))
                        toReturn += list(map(lambda x:x[0].toDetectionSample(timestamp=int(time.time()),
                                                                            value=x[1],
                                                                            detectorTag=detector.getTag(),
                                                                        baseModelTag=primaryModel.getTag(),
                                                                        maxLength=self.__maxDetectionLength), zipped))
                    except Exception as e:
                            return DetectionFailure(f'{detector.getTag()} had an exception "{e}" during detection with model: {primaryModel.getTag()} ',-1)
            #This too
            primaryModel.unload()

        return Right(toReturn)
        


    def addDetector(self, detector: BaseDetector) -> NoneResult:
        """
        Add a detector to the detection methods to be tested.
        """
        
        if not issubclass(detector.__class__, BaseDetector):
            return Left(DetectionFailure(f'Expected detector got: {detector.__class__.__name__}',-1))

        self.__detectors.append(detector)

        return Right(None)


    def registerPrimaryModel(self, model:DetectorMixin) -> NoneResult:
        """
        Inform the pipeline of the primary (base) model to be used by the detectors.
        """
        if not issubclass(model.__class__, (BaseModel, DetectorMixin)):
            return Left(DetectionFailure(f'Expected detector model got: {model.__class__.__name__}',-1))

        self.__primaryModels.append(model)

        return Right(None)


    def registerSecondaryModels(self, model:DetectorMixin, secondaryModels:list[DetectorMixin]) -> NoneResult:
        """
        Inform the pipeline of the valid secondary model combinations that are compatible with the primary (base) model.
        """
        if not issubclass(model.__class__, (BaseModel, DetectorMixin)):
            return Left(DetectionFailure(f'Expected primary detector model got: {model.__class__.__name__}',-1))

        for secondaryModel in secondaryModels:
            if not issubclass(secondaryModel.__class__, (BaseModel, DetectorMixin)):
                return Left(DetectionFailure(f'Expected secondary detector model got: {secondaryModel.__class__.__name__}',-1))

        self.__secondaryModels[model] = secondaryModels

        return Right(None)


    def setMaxDetectionLength(self, length:int|None) -> NoneResult:
        """
        Set the maximum amount of tokens a detector is allowed to use for detection from a sample.
        """
        self.__maxDetectionLength = length
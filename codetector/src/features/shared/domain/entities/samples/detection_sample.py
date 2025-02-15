from .sample import Sample
from dataclasses import dataclass
from abc import ABC


@dataclass(frozen=True, init=False)
class DetectionSample(Sample, ABC):
    """
    Immutable abstract class representing detection samples.
    """

    value: float
    """
    The value assigned by a detector to the sample.
    """

    detectorTag: str
    """
    The tag of the detector that assigned the value.
    """

    baseModelTag: str
    """
    The tag of the base model used.
    """

    secondaryModelTag: str|None = None
    """
    The tag of the secondary model used.
    """

    sampleHash: str
    """
    The hash of the sample that was analyzed.
    """

    promptHash: str
    """
    The hash of the prompt that was used to generate the sample.
    """

    maxLength: int|None = None
    """
    The max length that the detector used to assign value.
    """


    def __init__(self,
                content: str,
                prompt: str,
                originalPrompt: str,
                generatorTag: str,
                datasetTag: str,
                timestamp: int,
                value: float,
                detectorTag: str,
                baseModelTag: str,
                sampleHash: str,
                promptHash: str,
                topK: int | None = None,
                topP: float | None = None,
                temperature: float | None = None,
                secondaryModelTag: str | None = None,
                maxLength: int | None = None,):
        super().__init__(content, prompt, originalPrompt, generatorTag, datasetTag, timestamp, topK, topP, temperature)

        super().__setattr__('value',value)
        super().__setattr__('detectorTag',detectorTag)
        super().__setattr__('baseModelTag',baseModelTag)
        super().__setattr__('sampleHash', sampleHash)
        super().__setattr__('promptHash',promptHash)
        super().__setattr__('secondaryModelTag', secondaryModelTag if secondaryModelTag != 'None' else None)
        super().__setattr__('maxLength', maxLength if maxLength != 'None' else None)


    def toDetectionSample(self, timestamp, value, detectorTag, baseModelTag, secondaryModelTag = None, maxLength = None):
        return self
    

#kw_only=True necessary to get __init__ to work, requires __new__ hack
# class DetectionSample(_DetectionSample, ABC):
#     """
#     Immutable abstract class representing detection samples.
#     """

#     def __new__(mcls,content: str,
#             prompt: str,
#             originalPrompt: str,
#             generatorTag: str,
#             datasetTag: str,
#             timestamp: int,
#             value: float,
#             detectorTag: str,
#             baseModelTag: str,
#             sampleHash: str,
#             promptHash: str,
#             topK: int | None = None,
#             topP: float | None = None,
#             temperature: float | None = None,
#             secondaryModelTag: str | None = None,
#             maxLength: int | None = None):

#         return  _DetectionSample(content=content,
#                         prompt=prompt,
#                         originalPrompt=originalPrompt,
#                         generatorTag=generatorTag,
#                         datasetTag=datasetTag,
#                         timestamp=timestamp,
#                         topK=topK,
#                         topP=topP,
#                         temperature=temperature,
#                         value=value,
#                         detectorTag=detectorTag,
#                         baseModelTag=baseModelTag,
#                         secondaryModelTag=secondaryModelTag,
#                         sampleHash=sampleHash,
#                         promptHash=promptHash,
#                         maxLength=maxLength)

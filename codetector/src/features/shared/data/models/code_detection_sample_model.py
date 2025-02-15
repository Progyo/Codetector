from codetector.src.features.shared.data.models.mappable import MappableMixin
from codetector.src.features.shared.domain.entities.samples.detection_sample import DetectionSample
from dataclasses import dataclass

@dataclass(frozen=True, init=False)
class CodeDetectionSampleModel(DetectionSample, MappableMixin):
    """
    Model implementation of the detection sample class.
    """

    language : str
    """
    The language of the code sample that was analyzed.
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
                language: str,
                topK: int | None = None,
                topP: float | None = None,
                temperature: float | None = None,
                secondaryModelTag: str | None = None,
                maxLength: int | None = None,):
        super().__init__(content, prompt, originalPrompt, generatorTag, datasetTag, timestamp, value, detectorTag, baseModelTag, sampleHash, promptHash, topK, topP, temperature, secondaryModelTag, maxLength)
        super().__setattr__('language',language)


    def fromDict(sample:dict) -> 'CodeDetectionSampleModel':
        secondaryModel = sample['SecondaryModel']
        if secondaryModel == 'None':
            secondaryModel = None



        #Legacy support for samples that didn't store top k, top p, and temperature values
        topK = sample['TopK'] if 'TopK' in sample else "None"
        topP = sample['TopP'] if 'TopP' in sample else "None"
        temp = sample['Temperature'] if 'Temperature' in sample else "None"
        maxLen = sample['MaxLength'] if 'MaxLength' in sample else "None"
        lang = sample['Language'] if 'Language' in sample else None

        if topK != "None" and topK != None:
            topK = int(topK)
        else:
            topK = None

        if topP != "None" and topP != None:
            topP = float(topP)
        else:
            topP = None

        if temp != "None" and temp != None:
            temp = float(temp)
        else:
            temp = None

        if maxLen != "None" and maxLen != None:
            maxLen = int(maxLen)
        else:
            maxLen = None

        if lang == "None":
            lang = None

        #Temporary fix for detector bug outputing [value] instead of value
        try:
            value = float(sample['Value'])
        except ValueError:
            value = float(sample['Value'][1:-1])

        # CodeDetectionSampleModel(content,prompt, originalPrompt, generatorTag, datasetTag, timestamp, value, detectorTag, baseModelTag, sampleHash, promptHash,language, topK, topP, temperature, secondaryModelTag, maxLength,)
        return CodeDetectionSampleModel(sample['Code'],
                                        None,
                                        None,
                                        sample['Generator'],
                                        sample['Dataset'],
                                        int(sample['Date']),
                                        value,
                                        sample['Detector'],
                                        sample['BaseModel'],
                                        sample['CodeSampleHash'],
                                        sample['PromptHash'],
                                        lang,
                                        topK=topK,
                                        topP=topP,
                                        temperature=temp,
                                        secondaryModelTag=secondaryModel,
                                        maxLength=maxLen,
                                )
    
    def toDict(self, stringify:bool=True) -> dict:
        topK = self.topK
        if topK == None and stringify:
            topK = "None"

        topP = self.topP
        if topP == None and stringify:
            topP = "None"

        temp = self.temperature
        if temp == None and stringify:
            temp = "None"

        maxLen = self.maxLength
        if maxLen == None and stringify:
            maxLen = "None"

        return {
            'PromptHash': str(self.promptHash),
            'CodeSampleHash': str(self.sampleHash),
            'Language': str(self.language),
            'Value': str(self.value) if stringify else self.value,
            'MaxLength': str(self.maxLength) if stringify else self.maxLength,
            'Detector': self.detectorTag,
            'Generator': self.generatorTag,
            'BaseModel': self.baseModelTag,
            'SecondaryModel': str(self.secondaryModelTag),
            'Code': self.content,
            'TopK': str(topK) if stringify else topK,
            'TopP': str(topP) if stringify else topP,
            'Temperature': str(temp) if stringify else temp,
            'Date': str(self.timestamp) if stringify else self.timestamp,
            'Dataset': self.datasetTag,
        }
    

    def toDetectionSample(self, timestamp, value, detectorTag, baseModelTag, secondaryModelTag = None, maxLength = None) -> 'CodeDetectionSampleModel':
        return self
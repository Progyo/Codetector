from hashlib import sha256
from codetector.src.features.shared.data.models.mappable import MappableMixin
from codetector.src.features.shared.domain.entities.samples.code_sample import CodeSample
from codetector.src.features.shared.data.models.code_detection_sample_model import CodeDetectionSampleModel

class CodeSampleModel(CodeSample, MappableMixin):
    """
    Model implementation of the code sample class.
    """

    def fromLanguage(language:str) -> 'type[CodeSampleModel]':
        """
        Convert a string representation to a class instance of `CodeSampleModel`.
        """
        from .c import CSample
        from .cpp import CPPSample
        from .csharp import CSharpSample
        from .generic import GenericSample
        from .go import GoSample
        from .java import JavaSample
        from .javascript import JavascriptSample
        from .python import PythonSample
        from .rust import RustSample

        languages = {
            'python': PythonSample,
            'python3': PythonSample,
            'python2': PythonSample,
            'py': PythonSample,
            'javascript': JavascriptSample,
            'js': JavascriptSample,
            'c': CSample,
            'cpp': CPPSample,
            'c++': CPPSample,
            'java': JavaSample,
            'rust': RustSample,
            'go': GoSample,
            'c#': CSharpSample,
            'csharp': CSharpSample,
            'cs': CSharpSample,
        }

        if isinstance(language, str) and language in languages:
            return languages[language]
        else:
            return GenericSample
    

    def fromDict(sample:dict) -> 'CodeSampleModel':
        
        #Legacy support for samples that didn't store top k, top p, and temperature values
        topK = sample['TopK'] if 'TopK' in sample else "None"
        topP = sample['TopP'] if 'TopP' in sample else "None"
        temp = sample['Temperature'] if 'TopP' in sample else "None"

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

        try:
            date = int(sample['Date'])
        except ValueError:
            date = round(float(sample['Date']))

        #  CodeSample(content,prompt,originalprompt, generator, dataset, timestamp, topK, topP, temperature,)

        return CodeSampleModel.fromLanguage(sample['Language'])(sample['Code'],
                                                    sample['Prompt'],
                                                    sample['OriginalPrompt'],
                                                    sample['Generator'],
                                                    sample['Dataset'],
                                                    date,
                                                    topK,
                                                    topP,
                                                    temp,
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

        code = self.content
        if code == None and stringify:
            code = ''

        return {
            'Generator': self.generatorTag,
            'Language': self.getPL(),
            'Code': code,
            'TopK': str(topK) if stringify else topK,
            'TopP': str(topP) if stringify else topP,
            'Temperature': str(temp) if stringify else temp,
            'Prompt': str(self.prompt),
            'OriginalPrompt': str(self.originalPrompt),
            'Date': str(self.timestamp) if stringify else self.timestamp,
            'Dataset': self.datasetTag,
        }
    

    def __hash(self,text:str) -> int:
        """
        From: https://stackoverflow.com/questions/48613002/sha-256-hashing-in-python
        """
        hash = sha256(text.encode('utf-8')).hexdigest()
        return hash

    def toDetectionSample(self, timestamp, value, detectorTag, baseModelTag, secondaryModelTag = None, maxLength = None) -> CodeDetectionSampleModel:
        return CodeDetectionSampleModel(content=self.content,
                                        prompt=self.prompt,
                                        originalPrompt=self.originalPrompt,
                                        generatorTag=self.generatorTag,
                                        datasetTag=self.datasetTag,
                                        timestamp=timestamp,
                                        value=value,
                                        detectorTag=detectorTag,
                                        baseModelTag=baseModelTag,
                                        sampleHash=self.getHash(),
                                        promptHash=self.__hash(self.originalPrompt),
                                        language=self.getPL(),
                                        topK=self.topK,
                                        topP=self.topP,
                                        temperature=self.temperature,
                                        secondaryModelTag=secondaryModelTag,
                                        maxLength=maxLength
                                        )
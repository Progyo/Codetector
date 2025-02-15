
from codetector.src.core.failure import GenerateFailure
from codetector.src.core.typedef import NoneResult, Result
from codetector.src.features.shared.data.models.tokenizers.tiktoken import TikTokenTokenizer
from codetector.src.features.shared.domain.entities.models.base_model import BaseModel
from codetector.src.features.shared.domain.entities.models.generator_model import GeneratorMixin
from codetector.src.features.generation.domain.repositories import GeneratorRepository
from codetector.src.features.shared.domain.entities.samples.sample import Sample


from oslash import Right, Left

from codetector.src.features.shared.domain.entities.tokenizer import Tokenizer

try:
    import torch as torch
except ModuleNotFoundError:
    torch = None

class GeneratorRepositoryImplementation(GeneratorRepository):
    """
    Implementation of the generator repository.
    """

    def __init__(self):
        super().__init__()
        self.__generators : list[GeneratorMixin|BaseModel] = []
        """
        List of generators being managed.
        """
        self.__initialized : bool = False
        self.__tokenizer : Tokenizer = TikTokenTokenizer()
        """
        Tokenizer used for batch grouping.
        """


    def initialize(self) -> NoneResult:
        if not self.__initialized:
            #Do some initialization stuff
            for generator in self.__generators:
                try:
                    generator.initialize()
                    
                    #Set default values
                    # Needs to be patched in unit tests to prevent failures
                    # generator.setTopP(0.95)
                    # generator.setTopK(None)
                    # generator.setTemperature(0.97)
                    # generator.setMaxOutputLength(256)
                    # generator.setGenerateCount(1)
                    # generator.setBestOf(1,None)
                    # generator.enableDynamicSize(False)

                except Exception as e:
                    return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" during initialization',-1))
            self.__initialized = True

        return Right(None)


    def generateSingle(self, sample) -> Result[list[list[Sample]]]:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        if len(self.__generators) == 0:
            return Left(GenerateFailure('No generators added',-1))
        
        
        toReturn : list[list[Sample]] = []
        for generator in self.__generators:

            if not generator.supportsSample(sample):
                toReturn.append(None)
                generator.unload()
                continue

            #PyTorch inference optimizations (Disable gradient computation)
            if torch != None:
                with torch.inference_mode():
                    #Further inference optimizations
                    if generator.useSDP():
                        try:  
                            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                                toReturn.append(generator.generateSingle(sample))
                        except Exception as e:
                            return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while generating samples using SDP',-1))
            
                    else:
                        toReturn.append(generator.generateSingle(sample))
            else:
                toReturn.append(generator.generateSingle(sample))

            generator.unload()


        return Right(toReturn)


    def __groupBySize(self, samples:list[Sample], maxDiff:int|float) -> tuple[list[list[Sample]],list[list[int]]]:
        """
        Group samples by content size and maximum size difference threshold.
        This allows for better batching as less space is wasted in padding.
        Returns grouped list and list of original indices to return in correct order.
        maxDiff: When an int group by fixed threshold. When float, use relative percentage difference.
        """

        sortedSamples = sorted(map(lambda x: (x[1],len(self.__tokenizer.encode(x[1].content)),x[0]), enumerate(samples)),key=lambda x:x[1])
        
        output : list[list[Sample]] = []
        outputIndices : list[list[int]] = []

        smallest = 0

        for i, sample in enumerate(sortedSamples):
            if (isinstance(maxDiff,int) and sample[1] - sortedSamples[smallest][1] > maxDiff) or (
                isinstance(maxDiff,float) and sample[1] - sortedSamples[smallest][1] > sample[1] * maxDiff
            ):
                output.append(list(map(lambda x: x[0],sortedSamples[smallest:i])))
                outputIndices.append(list(map(lambda x: x[2],sortedSamples[smallest:i])))
                smallest = i
        
        output.append(list(map(lambda x: x[0],sortedSamples[smallest:])))
        outputIndices.append(list(map(lambda x: x[2],sortedSamples[smallest:])))

        return output, outputIndices

    def generateBatch(self, samples) -> Result[list[list[list[Sample]]]]:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        if len(self.__generators) == 0:
            return Left(GenerateFailure('No generators added',-1))
        
        grouped : tuple[list[list[Sample]],list[list[int]]] = self.__groupBySize(samples, 0.25)
        
        #Generator-Samples-GenerateCount
        generated : list[list[list[Sample]]] = [[] for _ in range(len(self.__generators))]

        for generatorIndex, generator in enumerate(self.__generators):
            for group in grouped[0]:
                #Filter out unsupported samples
                #Correct generator implementation should ignore None values in list
                #Should return None instead of list at index that was skipped
                # filteredGroup = [sample if generator.supportsSample(sample) else None for sample in group]

                filteredGroup : list[Sample] = []
                positions : list[int] = []
                for i,sample in enumerate(group):
                    if generator.supportsSample(sample):
                        filteredGroup.append(sample)
                        positions.append(i)

                #List of samples that were output
                output : list[list[Sample]] = []
                if torch != None:
                    with torch.inference_mode():
                        output = generator.generateBatch(filteredGroup)
                    torch.cuda.empty_cache()
                else:
                    output = generator.generateBatch(filteredGroup)

                #Add None to positions in generated[generatorIndex] where the sample was skipped
                added = 0
                for i in range(len(group)):
                    if i in positions:
                        generated[generatorIndex].append(output[added])
                        added+=1
                    else:
                        generated[generatorIndex].append(None)


            generator.unload()

        #Bring back into correct order

        #Flatten list
        correctIndices = [item for sublist in grouped[1] for item in sublist]

        toReturn : list[list[list[Sample]]] = [None for _ in range(len(self.__generators))]

        for generatorIndex, generatorSamples in enumerate(generated):
            toReturn[generatorIndex] = [None for _ in range(len(generatorSamples))]
            for sampleIndex, sample in enumerate(generatorSamples):
                toReturn[generatorIndex][correctIndices[sampleIndex]] = sample

        return Right(toReturn)

    def setTemperature(self, temperature) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))

        for generator in self.__generators:
            try:
                generator.setTemperature(temperature)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting temperature',-1))


        return Right(None)

    def setTopK(self, k) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.setTopK(k)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting topK',-1))

        return Right(None)
    
    def setTopP(self, p) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.setTopP(p)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting topP',-1))

        return Right(None)
    

    def setMaxOutputLength(self, length) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.setMaxOutputLength(length)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting max output length',-1))

        return Right(None)


    def setGenerateCount(self, count) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.setGenerateCount(count)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting generate count',-1))

        return Right(None)


    def addGenerator(self, generator) -> NoneResult:
        
        if not issubclass(generator.__class__, (BaseModel,GeneratorMixin)):
            return Left(GenerateFailure(f'Expected generator got: {generator.__class__.__name__}',-1))

        self.__generators.append(generator)
        
        return Right(None)

    def setBestOf(self, bestOf, heuristic) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.setBestOf(bestOf,heuristic)
            except Exception as e:
                return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while setting best of',-1))

        return Right(None)


    def enableDynamicSize(self, enable) -> NoneResult:
        if not self.__initialized:
            return Left(GenerateFailure('Generator not initialized',-1))
        for generator in self.__generators:
            try:
                generator.enableDynamicSize(enable)
            except Exception as e:
                if enable:
                    return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while disabling dynamic size',-1))
                else:
                    return Left(GenerateFailure(f'{generator.__class__.__name__} had an exception "{e}" while enabling dynamic size',-1))
        
        return Right(None)
                



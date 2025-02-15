from math import ceil
from codetector.samples import CodeSample, CodeDetectionSample
from codetector.detection import DetectorManager
from codetector.detection.usecases import (Initialize as InitializeUseCase,
                                            AddDetector as AddDetectorUseCase,AddDetectorParameters,
                                            Detect as DetectUseCase,DetectParameters,
                                            RegisterPrimaryModels as RegisterPrimaryModelsUseCase,RegisterPrimaryModelsParameters,
                                            RegisterSecondaryModels as RegisterSecondaryModelsUseCase,RegisterSecondaryModelsParameters)
from codetector import saveState,loadState

# Model imports
from models.codegeex2 import CodeGeeX2
from models.codegemma import CodeGemma_Instruct_7B
from models.codegen25 import CodeGen25_7B
from models.codellama import CodeLlama_7B,CodeLlama_Instruct_7B,CodeLlama_13B,CodeLlama_Instruct_13B
from models.incoder import Incoder_1B,Incoder_6B
from models.llama3 import Llama3_8B,Llama3_Instruct_8B
from models.phi import Phi_1B, Phi3Mini_Instruct_4B
from models.starcoder import StarCoder2_3B,StarCoder2_7B
from models.wavecoder import WaveCoderUltra_7B


# Detector imports
from detectors.loglikelihood import LogLikelihoodDetector
from detectors.entropy import EntropyDetector
from detectors.rank import RankDetector
from detectors.fastdetectgpt import FastDetectGPT
from detectors.binoculars import BinocularsDetector


from tqdm import tqdm


from codetector.dataset import  DatasetBatch

from dataset.generated_dataset import XMLGeneratedCodeDataset
from dataset.detection_dataset import XMLCodeDetectionDataset

from oslash import Right


if __name__ == '__main__':
    
    ### Resume state

    iterPath = 'detector_rng_state.pkl'
    iteration = loadState(iterPath)



    ### Use cases
    detector : DetectorManager = DetectorManager()

    Initialize = InitializeUseCase(detector)
    AddDetector = AddDetectorUseCase(detector)
    Detect = DetectUseCase(detector)
    RegisterPrimaryModels = RegisterPrimaryModelsUseCase(detector)
    RegisterSecondaryModels = RegisterSecondaryModelsUseCase(detector)


    ### Models
    phi1 = Phi_1B()
    geex2 = CodeGeeX2()
    codegen = CodeGen25_7B()
    phi3 = Phi3Mini_Instruct_4B()
    incoder = Incoder_1B()
    incoder6 = Incoder_6B()
    codellama7 = CodeLlama_7B()
    codellama13 = CodeLlama_13B()
    codellama7_instruct = CodeLlama_Instruct_7B()
    codellama13_instruct = CodeLlama_Instruct_13B()
    llama3_8 = Llama3_8B()
    llama3_8_instruct = Llama3_Instruct_8B()
    starcoder2_3 = StarCoder2_3B()
    starcoder2_7 = StarCoder2_7B()
    wavecoder = WaveCoderUltra_7B()
    codegemma = CodeGemma_Instruct_7B()

    
    ### Register models
    RegisterPrimaryModels(RegisterPrimaryModelsParameters([geex2,
                                                           codegen,
                                                           phi1,phi3,
                                                           incoder,incoder6,
                                                           codellama7,codellama13,
                                                           codellama7_instruct,codellama13_instruct,
                                                           llama3_8,llama3_8_instruct,
                                                           starcoder2_3,starcoder2_7,
                                                           wavecoder,
                                                           codegemma
                                                           ]))
    
    RegisterSecondaryModels(RegisterSecondaryModelsParameters({codellama13:[codellama13],
                                                               codellama13_instruct:[codellama13_instruct],
                                                               llama3_8:[llama3_8],
                                                               llama3_8_instruct:[llama3_8_instruct],
                                                               codellama7:[codellama13_instruct],
                                                               codellama7_instruct:[codellama13_instruct],
                                                               codegen:[codegen],
                                                               geex2:[geex2],
                                                               starcoder2_7:[starcoder2_7],
                                                               codegemma:[codegemma],
                                                               wavecoder:[wavecoder],
                                                               incoder6:[incoder6],
                                                               phi3:[phi3],
                                                               starcoder2_3:[starcoder2_7],
                                                               phi1:[phi1],
                                                               incoder:[incoder6]}))


    ### Detectors

    loglikelihood = LogLikelihoodDetector()
    AddDetector(AddDetectorParameters(loglikelihood))

    entropy = EntropyDetector()
    AddDetector(AddDetectorParameters(entropy))

    rank = RankDetector()
    AddDetector(AddDetectorParameters(rank))

    fastdetectgpt = FastDetectGPT(keepBothModelsLoaded=True)
    AddDetector(AddDetectorParameters(fastdetectgpt))

    binoculars = BinocularsDetector(keepBothModelsLoaded=True)
    AddDetector(AddDetectorParameters(binoculars))

    ### Datasets

    generatedDataset = XMLGeneratedCodeDataset(checkpointPath='data/generate.pkl')
    generatedDataset.loadDataset()

    finalDataset = XMLCodeDetectionDataset(checkpointPath='data/final.pkl')

    ### Detection parameters 


    batchSize = 8
    # When this is None, it is ignored
    maxLength = None


    ### Initialization
    
    Initialize()

    
    iteration = loadState(iterPath)

    #Progress bar
    bar : tqdm = tqdm(desc='Detecting samples',total=ceil(generatedDataset.getCount()/batchSize),position=0)

    batch : DatasetBatch = generatedDataset.loadBatch(batchSize)
    while not batch.final or len(batch.samples) > 0:
        result = Detect(DetectParameters(batch.samples))

        if isinstance(result, Right):
            for sample in result._value:
                finalDataset.addSample(sample)
                
        else:
            print(result._error.message)
            break
        
        iteration+=1
        saveState(iteration,iterPath)
        loadState(iterPath)
        generatedDataset.saveCheckpoint()


        batch = generatedDataset.loadBatch(batchSize)
        bar.update(1)

        finalDataset.save()
        finalDataset.saveCheckpoint()


    if not batch.final:
        finalDataset.save()
        finalDataset.saveCheckpoint()

    bar.close()


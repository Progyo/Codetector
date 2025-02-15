from math import ceil
from codetector.samples import CodeSample
from codetector.generation import GeneratorManager
from codetector.generation.usecases import (Initialize as InitializeUseCase,
                                            AddGenerator as AddGeneratorUseCase,AddGeneratorParameters,
                                            Generate as GenerateUseCase,GenerateParameters)
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


from tqdm import tqdm


from codetector.dataset import DatasetBatch
from dataset.stackoverflow import ParquetStackOverflowPostDataset,ParquetStackOverflowPreDataset
from dataset.apps import APPSDataset
from dataset.codesearchnet import CodeSearchNetPythonDataset
from dataset.leetcode_pre import LeetcodePreDataset
from dataset.leetcode_post import LeetCodePostDataset
from dataset.generated_dataset import XMLGeneratedCodeDataset

from codetector.dataset import AggregateDataset

from codetector.filters import PLFilter, DistributionFilter

from oslash import Right


if __name__ == '__main__':
    
    ### Resume state

    iterPath = 'generator_rng_state.pkl'
    iteration = loadState(iterPath)



    ### Use cases
    generator : GeneratorManager = GeneratorManager()

    Initialize = InitializeUseCase(generator)
    AddGenerator = AddGeneratorUseCase(generator)
    Generate = GenerateUseCase(generator)


    ### Models
    phi1 = Phi_1B()
    AddGenerator(AddGeneratorParameters(phi1))

    geex2 = CodeGeeX2()
    AddGenerator(AddGeneratorParameters(geex2))

    codegen = CodeGen25_7B()
    AddGenerator(AddGeneratorParameters(codegen))

    phi3 = Phi3Mini_Instruct_4B()
    AddGenerator(AddGeneratorParameters(phi3))

    incoder = Incoder_1B()
    AddGenerator(AddGeneratorParameters(incoder))

    incoder6 = Incoder_6B()
    AddGenerator(AddGeneratorParameters(incoder6))

    codellama7 = CodeLlama_7B()
    AddGenerator(AddGeneratorParameters(codellama7))

    codellama13 = CodeLlama_13B()
    AddGenerator(AddGeneratorParameters(codellama13))

    codellama7_instruct = CodeLlama_Instruct_7B()
    AddGenerator(AddGeneratorParameters(phi1))

    codellama13_instruct = CodeLlama_Instruct_13B()
    AddGenerator(AddGeneratorParameters(codellama13_instruct))

    llama3_8 = Llama3_8B()
    AddGenerator(AddGeneratorParameters(llama3_8))

    llama3_8_instruct = Llama3_Instruct_8B()
    AddGenerator(AddGeneratorParameters(llama3_8_instruct))

    starcoder2_3 = StarCoder2_3B()
    AddGenerator(AddGeneratorParameters(starcoder2_3))
    
    starcoder2_7 = StarCoder2_7B()
    AddGenerator(AddGeneratorParameters(starcoder2_7))

    wavecoder = WaveCoderUltra_7B()
    AddGenerator(AddGeneratorParameters(wavecoder))

    codegemma = CodeGemma_Instruct_7B()
    AddGenerator(AddGeneratorParameters(codegemma))

    ### Datasets


    f = PLFilter([CodeSample.fromLanguage('python'),
                   CodeSample.fromLanguage('java'),
                   CodeSample.fromLanguage('javascript'),
                   CodeSample.fromLanguage('csharp'),
                   CodeSample.fromLanguage('cpp'),
                   CodeSample.fromLanguage('rust'),
                   CodeSample.fromLanguage('go')])

    stackOverflow_pre = ParquetStackOverflowPreDataset(filters=[f,DistributionFilter('data/stackoverflow-pre_hash.pkl')])
    stackOverflow_post = ParquetStackOverflowPostDataset(filters=[f,DistributionFilter('data/stackoverflow-post_hash.pkl')])
    apps = APPSDataset(filters=[f,DistributionFilter('data/hf_apps_hash.pkl')])
    codeSearchNet = CodeSearchNetPythonDataset(filters=[f,DistributionFilter('data/hf_codesearchnet-python_hash.pkl')])
    leetCode_pre = LeetcodePreDataset(filters=[f,DistributionFilter('data/hf_leetcode-pre_hash.pkl')])
    leetCode_post = LeetCodePostDataset(filters=[f,DistributionFilter('data/leetcode-post_hash.pkl')])

    dataset = AggregateDataset([stackOverflow_pre,stackOverflow_post,
                                  apps,
                                  codeSearchNet,
                                  leetCode_pre,leetCode_post],checkpointPath='data/aggregate.pkl')

    generatedDataset = XMLGeneratedCodeDataset()
    
    dataset.loadDataset()

    ### Generation parameters 

    topP = 0.95
    temperature = 0.97
    batchSize = 8

    ### Initialization
    
    Initialize()

    iteration = loadState(iterPath)

    #Progress bar
    bar : tqdm = tqdm(desc='Generating samples',total=ceil(dataset.getCount()/batchSize),position=0)

    batch : DatasetBatch = dataset.loadBatch(batchSize)
    while not batch.final or len(batch.samples) > 0:
            # print(sample.getPL())
        result = Generate(GenerateParameters(samples=batch.samples,
                                             batch=True,
                                             temperature=temperature,
                                             topP=topP,
                                             dynamicSize=True,
                                             generateCount=1))

        if isinstance(result, Right):
            for i in range(len(batch.samples)):
                #Interlace human samples in generated dataset
                generatedDataset.addSample(batch.samples[i])
                #Add all generator 
                for generatorOutput in result._value:                    
                    for samples in generatorOutput:
                        if isinstance(samples, list):
                            generatedDataset.addSample(samples[i])
                        else:
                            generatedDataset.addSample(samples)
        else:
            print(result._error.message)
            break
        
        iteration+=1
        saveState(iteration,iterPath)
        loadState(iterPath)
        dataset.saveCheckpoint()


        batch = dataset.loadBatch(batchSize)
        bar.update(1)
        generatedDataset.save()
        generatedDataset.saveCheckpoint()

    if not batch.final:
        generatedDataset.save()
        generatedDataset.saveCheckpoint()
    bar.close()


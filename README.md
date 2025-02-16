# Codetector

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Status](https://img.shields.io/badge/status-active-success.svg)](/)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [License](#license)

## Description

Codetector is a modular AI-generated content (AIGC) detection framework that allows easy integration of custom datasets, large language models (LLMs), and detection methods. With features like crash recovery and unit-tested code, it provides a flexible and robust pipeline for AIGC generation and detection. Initially designed for code detection, we developed it to produce a large corpus of 113,776 code samples and 12,294,388 detection samples using 17 LLMs and five detection methods. You can read more about the process and results we produced in our paper titled "Codetector: A Framework for Zero-shot Detection of AI-Generated Code".

## Features

- Modular: The framework is designed to be flexible and accomodate changes in any layer of the pipeline. Don't like the existing `run_generation_pipeline.py` or `run_detection_pipeline.py` file? Then change it to your heart's content and utilise the the underlying use cases and managers as you wish.
- Expandable: Easily integrate new dataset/sources, LLMs, and detection methods in the pipeline. Simply implement/extend the necessary abstract classes and mixins defined in the framework and reference your implementation in the pipeline files.
- Crash Recovery: Crashes in the pipeline can be resumed from the latest batch. Data in the batch will be lost as states are only saved after each batch.
- Reproducibility: The same starting parameters and pipeline configuration will result in the same output when running on the same machine. It has not been tested for other machines yet but should work in theory.
- Tested: The framework is unit tested (with more tests on the way) to increase confidence in its underlying functionality. It is up to the end user to ensure that the LLMs they add and use are configured and implemented properly.

## Installation

1.  Clone the repository:
    ```
    git clone https://github.com/Progyo/Codetector
    ```

We provide several ways to use our framework. Note that some LLMs may have their own dependencies. For the framework itself, depending on your preferences, choose one of the following ways to install the dependencies:

### Conda
For full control of packages being installed, follow these step-by-step instructions on how to install the project using conda.


2. Create conda environment using `conda create --name codetector python=3.10`.
3. :exclamation: You **must** patch `typing.py` to support generic typedefs in Python. You may run `patcher.py` at your on discretion to automatically patch your `typing.py` file. Beware that the script modifies base Python packages and may break your Python installc else:
   1. Locate the `typing.py` file used by your conda environment. It is usually located under `lib/python3.10/typing.py`.
   2. Replace the class `NewType` in `typing.py` with the code found [here](https://gist.github.com/eltoder/4035faa041112a988dcf3ab101fb3db1).
4. For the general framework, install the following (or run `pip install -r requirements.txt`):
   1. For progress bars, install **tqdm** using `pip install tqdm`
   2. For general math operation support, install **numpy** using `pip install numpy`
   3. For monad support, install **oslash** using `pip install oslash`
   4. For DataFrame support, install **pandas** using `pip install pandas`
   5. For Apache Parquet dataset file format support, install **pyarrow** using `pip install pyarrow`
   6. For Hugging Face Datasets as dataset provider support, install **datasets** using `pip install datasets`
   7. For NLTKTokenizer and TikTokenTokenizer support, install **nltk** and **tiktoken** using `pip install nltk tiktoken`
   8. For plotting graphs and other figures, install **matplotlib** using `pip install matplotlib`
   9. For calculating ROC and other metrics, install **scikit-learn** using `pip install scikit-learn` 
5. To add CUDA support, install the following:
   1. Install **cudatoolkit** using `conda install cudatoolkit=11.8.0`
   2. Install **cuda-nvcc 11.8** using `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`
   3. Install **PyTorch** using `conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
6. To add Hugging Face Transformers LLM inference support (with quantization), install the following:
   1. Install **transformers** and **optimum** using `pip install transformers>=4.32.0 optimum>=1.12.0`
   2. Install **ctransformers** using `pip install ctransformers[cuda]>=0.2.24`
   3. Install **bitsandsbytes** using `pip install bitsandbytes`
7. Ensure PyTorch with CUDA is installed
   1. Install **PyTorch** using `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`
8. If using OpenAI API:
   1. For API access to OpenAI (for labelling or generation), install **openai** using `pip install openai`
   2. Set the following environment variables:
      1. **OPENAI_API_KEY**: `conda env config vars set OPENAI_API_KEY=<YOUR API KEY>`
      2. **OPENAI_ORG_ID**: `conda env config vars set OPENAI_ORG_ID=<YOUR ORG ID>`


### Docker

:warning: We are currently working on supplying a docker file / image :construction:


## Datasets

The datasets from various stages of our generation and detection process can be found below. Alongside the datasets, we supply hashlists stored in Python Pickle files that can be used to filter them to match distributions that we used in our paper. Place all of the hash list files in the `data` folder in the root directory. It is important to note that the individual datasets themselves are not part of the framework. They can be located outside in the `dataset` folder in the root directory. Feel free to add your own implementations in this folder (see [Custom Dataset](#custom-dataset)).

### Source

Source datasets exclusively contain human-generated code samples. We used a mixture of existing datasets and newly collected ones. In the following we describe how to download and setup the individual datasets.

#### APPS and CodeSearchNet

For APPS and CodeSearchNet, we utilise [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index). Ensure that you have installed the dependency with `pip install datasets`. Then simply import the dataset. You can find the hash lists [here]().

Usage:
```py
from dataset.apps import APPSDataset
from dataset.codesearchnet import CodeSearchNetPythonDataset
from codetector.filters import DistributionFilter

# Define the datasets
apps = APPSDataset(filters=[DistributionFilter('data/hf_apps_hash.pkl')])
codesearch = CodeSearchNetPythonDataset(filters=[DistributionFilter('data/hf_codesearchnet-python_hash.pkl')])

# Load the datasets
apps.loadDataset()
codesearch.loadDataset()

# Load a batch of 10 samples
batch = apps.loadBatch(10)

# Loop through the samples and print the content
for sample in batch.samples:
   print(sample.content)

```

#### LeetCode

For LeetCode we utilise two sources. For the "Pre" dataset we utilise [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index), like APPS and CodeSearchNet. For the "Post" dataset we use our own web-scraped dataset and extend the `Dataset` abstract class defined in the framework to implement JSON reading capabilities for the specific dataset. You can find the hash lists [here]().

For LeetCode Post, you must download the dataset from [here]() and place it under `data/leetcode_post/samples.json`.

Usage:
```py
from dataset.leetcode_pre import LeetcodePreDataset
from dataset.leetcode_post import LeetCodePostDataset
from codetector.filters import DistributionFilter

# Define the datasets
leetcodePre = LeetcodePreDataset(filters=[DistributionFilter('data/hf_leetcode-pre_hash.pkl')])
leetcodePost = LeetCodePostDataset(filters=[DistributionFilter('data/leetcode-post_hash.pkl')])

...
```

#### Stack Overflow

For Stack Overflow, like LeetCode, we split the dataset into "Pre" and "Post". We supply the split datasets individually as two compressed Apache Parquet partition datasets. You can download the Pre dataset [here]() and the Post dataset [here](). Place the `.parquet` files in `data/stackoverflow_pre` and `data/stackoverflow_post` respectively. You can find the hash lists [here]().

Usage:
```py
from dataset.stackoverflow import ParquetStackOverflowPreDataset, ParquetStackOverflowPostDataset
from codetector.filters import DistributionFilter

# Define the datasets
stackoverflowPre = ParquetStackOverflowPreDataset(filters=[DistributionFilter('data/stackoverflow-pre_hash.pkl')])
stackoverflowPost = ParquetStackOverflowPostDataset(filters=[DistributionFilter('data/stackoverflow-post_hash.pkl')])

...
```


### Generated and Detection

The generation and detection pipelines require the generated and detection samples to be saved to another dataset. For ease of use, we implemented a generated and detection dataset class (Specifically for CodeSamples and CodeDetectionSamples).

Usage:
```py
from dataset.generated_dataset import ParquetGeneratedCodeDataset
from dataset.detection_dataset import ParquetCodeDetectionDataset

# Define the datasets
generated = ParquetGeneratedCodeDataset()
detection = ParquetCodeDetectionDataset()

...
```

If you would prefer these datasets to be human-readable at the cost of file size, you can use the XML equivalent classes:
```py
from dataset.generated_dataset import XMLGeneratedCodeDataset
from dataset.detection_dataset import ParquetCodeDetectionDataset

# Define the datasets
generated = XMLGeneratedCodeDataset()
detection = XMLCodeDetectionDataset()

...
```

### Aggregate Dataset

The aggregate dataset serves as a way to easily merge datasets together and treat them as one.

Usage:
```py
from codetector.dataset import AggregateDataset
from dataset.stackoverflow import ParquetStackOverflowPreDataset, ParquetStackOverflowPostDataset
from codetector.filters import DistributionFilter

# Define the datasets
stackoverflowPre = ParquetStackOverflowPreDataset(filters=[DistributionFilter('data/stackoverflow-pre_hash.pkl')])
stackoverflowPost = ParquetStackOverflowPostDataset(filters=[DistributionFilter('data/stackoverflow-post_hash.pkl')])

stackoverflow = AggregateDataset([stackoverflowPre,stackoverflowPost])

...
```

### Custom Dataset

There are two levels at which one can implement their own dataset/source at. The easiest way is to simply extend and implement the necessary abstract methods of an existing dataset format/source e.g.: `XMLDataset`, `ParquetDataset`, `HuggingFaceDataset`. Note that each existing format/source may have its own additional methods that need to be implemented. The second option is to implement the abstract `Dataset` class that gives you the most flexibility.

Option 1:
```py
from codetector.dataset import XMLDataset
from codetector.samples import CodeSample

class CustomXML(XMLDataset):

    def getContentType(self):
        #Define the object type contained in the dataset.
        #This allows for serialisation into arbitrary file formats.
        return CodeSample

    def preProcess(self):
        #Called when loading a dataset for the first time.
        pass

    def getTag(self):
        #The tag of the dataset.
        return 'custom_xml'
    
xml = TestXML('data/sample_data')
```

Option 2:
```py
from codetector.dataset.abstract import Dataset, Sample

class CustomDataset(Dataset):
   #Implementation
   ...
```

## Figures

All our figures can be downloaded [here](). Optionally, you can also generate them yourself, using the supplied `.ipynb` notebooks in the `notebooks` folder.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
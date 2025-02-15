# Codetector
This repository contains all the code for the paper titled "Codetector: A Framework for Zero-shot Detection of AI-Generated Code".

## Setup
The code has been tested in a conda environment running Python 3.10.13 on Ubuntu using WSL-2.

1. Create conda environment using `conda create --name codetector python=3.10`.
2. To add CUDA support, install the following:
   1. Install **cudatoolkit** using `conda install cudatoolkit=11.8.0`
   2. Install **cuda-nvcc 11.8** using `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`
   3. Install **PyTorch** using `conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
3. To add LLM inference support (with quantization), install the following:
   1. Install **transformers** and **optimum** using `pip install transformers>=4.32.0 optimum>=1.12.0`
   //2. Install **auto-gptq** using `pip install auto-gptq==0.6.0 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`
   3. Install **ctransformers** using `pip install ctransformers[cuda]>=0.2.24`
   4. (CodeGeeX2) `pip install protobuf cpm_kernels (gradio mdtex2html) sentencepiece accelerate`
   5. Install **bitsandsbytes** using `pip install bitsandbytes`
4. For the dataset creation pipeline, install the following:
   1. For progress bars, install **progress** using `pip install progress`
   2. For automatic labeling of code, install **openai** using `pip install openai`
   3. For parsing of xml documents, install **beautifulsoup4** and **lxml** using `pip install beautifulsoup4 lxml`
   4. For getting token length estimates, install **tiktoken** using `pip install tiktoken`
   5. For plotting graphs and other figures, install **matplotlib** using `pip install matplotlib`
   6. For calculating ROC and other metrics, install **scikit-learn** using `pip install scikit-learn` 
5. For increased inference speed, install the following:
   1. Install **flash-attn** using `pip install flash-attn`
   //2. Install **optimum-nvidia*** using `python -m pip install --pre --extra-index-url https://pypi.nvidia.com optimum-nvidia`
   //   1. May require running `apt-get install -y openmpi-bin libopenmpi-dev` and/or `conda install mpi4py` beforehand
6. Ensure PyTorch with cuda is installed
   1. Install **PyTorch** using `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`
7. If using Open-AI to label, set the following environment variables:
   1. **OPENAI_API_KEY**: `conda env config vars set OPENAI_API_KEY=<YOUR API KEY>`
   2. **OPENAI_ORG_ID**: `conda env config vars set OPENAI_ORG_ID=<YOUR ORG ID>`

## Docker
WIP

## Framework Details
WIP

## Results
WIP
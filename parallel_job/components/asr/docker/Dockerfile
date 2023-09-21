# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN pip install 'faster-whisper==0.8.0 ' \
                'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'soundfile==0.12.1' \
                'cloudpickle==1.1.1' \
                'librosa==0.10.1' \
                'num2words==0.5.12'
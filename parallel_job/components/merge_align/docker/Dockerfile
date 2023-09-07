# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN pip install 'deepmultilingualpunctuation==1.0.1' \
                'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]' \
                'azureml-telemetry' \
                'azureml-core'
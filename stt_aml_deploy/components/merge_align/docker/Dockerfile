# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN pip install 'transformers==4.35.0' \
                'protobuf==3.20.0' \
                'pyyaml==5.4.1' \
                'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'azure-identity==1.15.0' \
                'azure-keyvault-secrets==4.7.0' \
                'pgpy==0.6.0' \
                'pymongo==4.6.1'
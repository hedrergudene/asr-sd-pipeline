# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'faster-whisper==0.9.0'
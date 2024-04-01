# Base image
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.1-cuda12.1:6

RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'azure-identity==1.15.0' \
                'azure-keyvault-secrets==4.7.0' \
                'pgpy==0.6.0' \
                'faster-whisper==1.0.1'
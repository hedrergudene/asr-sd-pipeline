# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.01

# Install pip dependencies
## Diarization
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]' \
                'azureml-telemetry' \
                'azureml-core'
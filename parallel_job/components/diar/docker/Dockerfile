# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.06

# Install pip dependencies
## Diarization
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'pyyaml==5.4.1' \
                'fire==0.5.0'
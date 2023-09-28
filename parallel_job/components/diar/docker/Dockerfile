# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.06

# Install pip dependencies
## Diarization
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'mltable>=1.2.0' \
                'azureml-dataset-runtime[pandas,fuse]==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-core==1.53.0' \
                'numpy==1.24.4' \
                'cloudpickle==1.1.1' \
                'soundfile==0.12.1' \
                'librosa==0.10.1' \
                'audiofile==1.3.0'
# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.01

# Install pip dependencies
## Diarization
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'azureml-core==1.47.0' \
                'azureml-defaults==1.47.0' \
                'azure-identity==1.12.0' \
                'soundfile==0.12.1' \
                'cloudpickle==1.1.1' \
                'librosa==0.10.1' \
                'fire==0.5.0'
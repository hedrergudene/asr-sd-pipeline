# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.06

# Install pip dependencies
## Diarization
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'azureml-core==1.47.0' \
                'azure-ai-ml==1.10.1' \
                'numpy==1.24.4' \
                'cloudpickle==1.1.1' \
                'librosa==0.10.1' \
                'soundfile==0.12.1' \
                'audiofile==1.3.0' \
                'fire==0.5.0'
# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'azureml-core==1.47.0' \
                'azure-ai-ml==1.10.1' \
                'git+https://github.com/m-bain/whisperx.git ' \
                'soundfile==0.12.1' \
                'cloudpickle==1.1.1' \
                'librosa==0.10.1' \
                'audiofile==1.3.0' \
                'num2words==0.5.12' \
                'pandas==2.0.3' \
                'numpy==1.26.0' \
                'fire==0.5.0'
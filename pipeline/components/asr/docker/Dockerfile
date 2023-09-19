# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
## ASR
RUN pip install 'azureml-core==1.47.0' \
                'azureml-defaults==1.47.0' \
                'azure-identity==1.12.0' \
                'faster-whisper==0.9.0 ' \
                'soundfile==0.12.1' \
                'cloudpickle==1.1.1' \
                'librosa==0.10.1' \
                'fire==0.5.0'
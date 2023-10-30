# Base image (check CUDA version is the same as PyTorch one)
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
RUN python -m pip install 'azureml-core==1.47.0' \
                          'azure-ai-ml==1.10.1' \
                          'demucs==4.0.1' \
                          'protobuf==3.20.0' \
                          'pyyaml==5.4.1' \
                          'fire==0.5.0'
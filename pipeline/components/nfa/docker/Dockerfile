# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.06

# Install pip dependencies
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN pip install 'azureml-core==1.47.0' \
                'azure-ai-ml==1.10.1' \
                'pyyaml==5.4.1' \
                'fire==0.5.0'
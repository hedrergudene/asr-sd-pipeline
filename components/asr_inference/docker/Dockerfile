# Base image (check curated environments https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install pip dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
RUN pip install 'faster-whisper==0.6.0' \
                'git+https://github.com/m-bain/whisperx.git' \
                'soundfile==0.12.1' \
                'omegaconf==2.2.3' \
                'pandas==1.3.5' \
                'fire==0.5.0'
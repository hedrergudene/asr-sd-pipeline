# Base image (check curated environments https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment)
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

# Install pip dependencies
RUN pip install 'openai-whisper==20230314' \
                'stable-ts==2.0.1' \
                'git+https://github.com/m-bain/whisperX.git@4cb167a225c0ebaea127fd6049abfaa3af9f8bb4' \
                'omegaconf==2.3.0' \
                'pandas==1.3.5' \
                'librosa==0.10.0.post2' \
                'soundfile==0.12.1'
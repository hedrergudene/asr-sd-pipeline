# Base image (check curated environments https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment)
FROM python:3.11.4-slim

# Install dependencies
RUN pip install 'azure-identity==1.12.0' \
                'azure-storage-blob==12.13.0' \
                'protobuf==3.20.0' \
                'pandas==1.3.5' \
                'pyyaml==5.4.1' \
                'fire==0.4.0' 
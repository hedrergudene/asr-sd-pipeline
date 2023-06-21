# Base image (check curated environments https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment)
FROM python:3.11.4-slim

# Install dependencies
RUN pip install 'azureml-core==1.47.0' \
                'azureml-defaults==1.47.0' \
                'azure-identity==1.12.0' \
                'azure-storage-blob==12.13.0' \
                'deepmultilingualpunctuation==1.0.1' \
                'protobuf==3.20.0' \
                'pyyaml==5.4.1' \
                'fire==0.4.0'
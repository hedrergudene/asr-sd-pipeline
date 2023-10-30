# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.06

# Install pip dependencies
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg
RUN python -m pip install 'azureml-core==1.47.0' \
                'azure-keyvault-secrets==4.7.0' \
                'azure-identity==1.14.1' \
                'azure-ai-ml==1.10.1' \
                'transformers==4.33.1' \
                'optimum==1.13.2' \
                'accelerate==0.23.0' \
                'onnxruntime==1.16.1' \
                'demucs==4.0.1' \
                'submitit==1.5.0' \
                'cloudpickle==1.5.0' \
                'audiofile==1.3.0' \
                'protobuf==3.20.3' \
                'pyyaml==5.4.1' \
                'fire==0.5.0'
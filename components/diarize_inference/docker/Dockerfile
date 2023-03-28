# Base image (check CUDA version is the same as PyTorch one)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install pip dependencies
RUN pip install 'nemo_toolkit[asr]==1.15.0' \
                'omegaconf==2.3.0' \
                'pandas==1.3.5' \
                'librosa==0.10.0.post2' \
                'soundfile==0.12.1'
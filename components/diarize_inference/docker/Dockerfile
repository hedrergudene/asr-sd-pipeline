# Base image (check CUDA version is the same as PyTorch one)
FROM nvcr.io/nvidia/nemo:23.01

# Install pip dependencies
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN pip install 'omegaconf==2.2.3' \
                'pandas==1.3.5' \
                'librosa==0.10.0.post2' \
                'fire==0.5.0'
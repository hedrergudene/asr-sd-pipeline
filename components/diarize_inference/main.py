# Requierments
import logging as log
import os
import time
import shutil
import subprocess
from io import BytesIO
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import re
import json
import sys
from pathlib import Path
import librosa
import torch
import fire
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR


# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Helper function to fetch files
def get_file(f):
    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            log.error(f"More than one file was found in directory: {','.join(files)}.")
            return (f"More than one file was found in directory: {','.join(files)}.", 500)

# Helper function to get CUDA compute capability
def get_cuda_compute():
    output = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"], 
                            stdout=subprocess.PIPE, 
                            text=True
    )
    return float(output.stdout.split('\n')[1])

# Helper function to build NeMo input manifest
def create_msdd_config(audio_filenames:List[str]):
    with open("input/nemo_manifest.jsonl", "w") as fp:
        for x in audio_filenames:
            json.dump({
                    "audio_filepath": x,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "rttm_filepath": None,
                    "uem_filepath": None,
               }, fp)
            fp.write('\n')


# Helper function to process diarization output
def process_NeMo_output(diar_output):
    return pd.DataFrame([{'start':float(x.split(' ')[0]), 'end': float(x.split(' ')[1]), 'speaker':x.split(' ')[2][-1]} for x in diar_output])



# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    input_transcriptions,
    event_type,
    max_num_speakers,
    word_ts_anchor_offset,
    output_path
):
    # Create output path and auxiliary folders
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path('./nemo_output').mkdir(parents=True, exist_ok=True)
    Path('./transcriptions').mkdir(parents=True, exist_ok=True)
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Check cuda capabilities
    if ((torch.cuda.is_available()) & (get_cuda_compute()<7.5)):
        raise Exception(f"Nvidia CUDA compute capabilities are below 7.5 ({get_cuda_compute()}), threshold for Turing tensor cores.")
    # Fetch input paths from previous component
    input_filepath = get_file(input_path)
    os.system(f"unrar e {get_file(input_transcriptions)} ./transcriptions")
    pathdirs = pd.read_csv(input_filepath)['paths'].values
    storage_id = re.findall('^https://(.*?).blob.core.windows.net/',pathdirs[0])[0]
    container_id = re.findall(f'^https://{storage_id}.blob.core.windows.net/(.*?)/',pathdirs[0])[0]
    # Get container client to get files, as librosa cannot fetch them directly
    account_url = f"https://{storage_id}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container=container_id)
    log.info(f"Preprocessing loop:")
    for filepath in pathdirs:
        # Read
        log.info(f"\tProcessing file: {filepath}")
        filename = re.findall(f'^https://{storage_id}.blob.core.windows.net/{container_id}/(.*?)$',filepath)[0]
        filename, extension = os.path.splitext(filename)
        signal, sample_rate = librosa.load(BytesIO(container_client.download_blob(f"{filename}{extension}").readall()), sr=None) # load audio from storage
        # Convert file to 16kHz mono
        log.info(f"\tConvert to 16kHz mono")
        prep_time = time.time()
        if sample_rate!=16000:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
        if len(signal.shape)>1:
            signal = librosa.to_mono(signal)
        filename = filepath.split('/')[-1].split('.')[0] # get id
        librosa.output.write_wav(f'./input_audios/{filename}{extension}', signal, 16000) # save in tmp path as 16kHz, mono
        prep_time = time.time() - prep_time
        log.info(f"\t\tPrep. time: {prep_time}")
    # Fetch word-level timestamps from ASR
    log.info(f"Fetch word-level timestamps from ASR")
    filenames = os.listdir('./input_audios')
    filenames = [os.path.splitext(x)[0] for x in filenames]
    idx2dfs = {x:pd.read_csv(os.path.join(f'./transcriptions/{filename}.csv')) for x in filenames}
    word_ts = {k:[[row.start, row.end] for row in v.itertuples(index=False)] for k,v in idx2dfs.items()}
    # Read cfg
    log.info(f"Read NeMo MSDD configuration file:")
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{event_type}.yaml')
    msdd_cfg.clustering.max_num_speakers = max_num_speakers
    create_msdd_config(os.listdir('./input_audios')) # initialise msdd cfg
    # NeMo MSDD
    log.info(f"Initialize NeMo MSDD diarization model:")
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)
    msdd_model.audio_file_list = [f"./input_audios/{x}" for x in os.listdir('./input_audios')]
    msdd_model.word_ts_anchor_offset = word_ts_anchor_offset
    msdd_model.cfg_diarizer.asr.parameters.asr_based_vad = True
    # Diarization
    log.info(f"Run diarization")
    diar_time = time.time()
    diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, word_ts)
    diar_time = time.time() - diar_time
    log.info(f"\tDiarization time: {diar_time}")
    # Process diarization output
    log.info(f"Save outputs")
    for filename, x in diar_hyp:
        process_NeMo_output(x).to_csv(os.path.join(f'./outputs/{filename}.csv'), index=False)
    shutil.rmtree(f'./input_audios') # remove audio
    # Generate output
    os.system(f"rar a {os.path.join(output_path,'diars.rar')} ./outputs")

if __name__=="__main__":
    fire.Fire(main)
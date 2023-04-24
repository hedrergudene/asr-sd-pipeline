# Requierments
import logging as log
import os
import subprocess
from io import BytesIO
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
def create_msdd_config(audio_filename:str):
    meta = {
        "audio_filepath": audio_filename,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open("input/nemo_manifest.json", "w") as fp:
        json.dump(meta, fp)


# Helper function to process diarization output
def process_NeMo_output(diar_output):
    return pd.DataFrame([{'start':float(x.split(' ')[0]), 'end': float(x.split(' ')[1]), 'speaker':x.split(' ')[2][-1]} for x in diar_output])



# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    input_transcriptions,
    event_type,
    word_ts_anchor_offset,
    output_path
):
    # Create output path and auxiliary folders
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
    # Read cfg
    log.info(f"Read NeMo MSDD configuration file:")
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{event_type}.yaml')
    create_msdd_config(f'./input/dummy_audio.wav') # initialise msdd cfg
    # NeMo MSDD
    log.info(f"Initialize NeMo MSDD diarization model:")
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)
    msdd_model.word_ts_anchor_offset = word_ts_anchor_offset
    log.info(f"Inference loop:")
    for filepath in pathdirs:
        # Read
        log.info(f"\tProcessing file: {filename}")
        filename = re.findall(f'^https://{storage_id}.blob.core.windows.net/{container_id}/(.*?)$',filepath)[0]
        filename, extension = os.path.splitext(filename)
        signal, sample_rate = librosa.load(BytesIO(container_client.download_blob(f"{filename}{extension}").readall()), sr=None) # load audio from storage
        # Convert file to 16kHz mono
        log.info(f"\tConvert to 16kHz mono")
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
        signal = librosa.to_mono(signal)
        filename = filepath.split('/')[-1].split('.')[0] # get id
        librosa.output.write_wav(f'./input/{filename}{extension}', signal, 16000) # save in tmp path as 16kHz, mono
        # Adapt model config
        log.info(f"\tAdapt model config")
        create_msdd_config(f'./input/{filename}.wav') # adapt cfg
        msdd_model.audio_file_list = [f'{filename}.wav']
        # Fetch word-level timestamps from ASR
        log.info(f"\tFetch word-level timestamps from ASR")
        df = pd.read_csv(os.path.join(f'./transcriptions/{filename}.csv'))
        word_ts = {filename:[[row.start, row.end] for row in df.itertuples(index=False)]}
        # Diarization
        log.info(f"\tRun diarization")
        msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)
        msdd_model.word_ts_anchor_offset = 0
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, word_ts)
        # Process diarization output
        log.info(f"\tSave output")
        diar_df = process_NeMo_output(diar_hyp[filename])
        diar_df.to_csv(os.path.join(f'./outputs/{filename}.csv'), index=False)
        os.remove(f'./input/{filename}{extension}') # remove audio
    # Generate output
    os.system(f"rar a {os.path.join(output_path,'diars.rar')} ./outputs")

if __name__=="__main__":
    fire.Fire(main)
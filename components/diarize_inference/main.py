# Requierments
import logging as log
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import re
import json
import sys
from pathlib import Path
import librosa
import soundfile
import fire
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer


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
def process_NeMo_output(filename:str):
    # Open output
    with open(f'./nemo_output/pred_rttms/{filename}.rttm', 'r') as f:
        lines = f.read().split('\n')
    # Iterate through lines
    diar_list = []
    for line in lines:
        try:
            start, offset = [float(x) for x in re.findall('\d+[.]\d{3}', line)]
            speaker = int(re.findall('<NA>\s{1}speaker_(.*?)\s{1}<NA>', line)[0])
            diar_list.append({'start':start, 'end': start+offset, 'speaker':speaker})
        except:
            continue
    # Save csv file
    pd.DataFrame(diar_list).to_csv(f'./nemo_output/pred_csv/{filename}.csv')



# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    event_type,
    output_path
):
    # Create output path and auxiliary folders
    Path('./nemo_output/pred_csv').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Fetch input paths from previous component
    input_filepath = get_file(input_path)
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
    msdd_cfg = OmegaConf.load(f'./input/ClusterDiarizer_{event_type}.yaml')
    create_msdd_config(f'./input/dummy_audio.wav') # initialise msdd cfg
    # NeMo MSDD
    log.info(f"Initialize NeMo MSDD diarization model:")
    msdd_model = NeuralDiarizer(cfg=msdd_cfg)
    log.info(f"Inference loop:")
    for filepath in pathdirs:
        filename = re.findall(f'^https://{storage_id}.blob.core.windows.net/{container_id}/(.*?)$',filepath)[0]
        filename, extension = os.path.splitext(filename)
        signal, sample_rate = librosa.load(BytesIO(container_client.download_blob(f"{filename}{extension}").readall()), sr=None) # load audio from storage
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
        signal = librosa.to_mono(signal)
        filename = filepath.split('/')[-1].split('.')[0] # get id
        librosa.output.write_wav(f'./input/{filename}{extension}', signal, 16000) # save in tmp path as 16kHz, mono
        create_msdd_config(f'./input/{filename}{extension}') # adapt cfg
        msdd_model.diarize() # output lies in './nemo_output' folder
        process_NeMo_output(filename)
        os.remove(f'./input/{filename}{extension}') # remove audio
    # Generate output
    os.system(f"rar a {os.path.join(output_path,'diars.rar')} ./nemo_output/pred_csv")

if __name__=="__main__":
    fire.Fire(main)
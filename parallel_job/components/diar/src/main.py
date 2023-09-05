# Libraries
import argparse
import sys
import logging as log
import requests
import soundfile as sf
import librosa
from pathlib import Path
import os
import re
import time
import subprocess
from typing import List
import json
import torch
import pandas as pd
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Helper function to build NeMo input manifest
def create_msdd_config(audio_filenames:List[str]):
    if os.path.exists("input/nemo_manifest.jsonl"): os.remove("input/nemo_manifest.jsonl")
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
def process_NeMo_output(output_path):
    out = []
    for x in os.listdir(os.path.join(output_path, 'pred_rttms')):
        segs = []
        filename, extension = os.path.splitext(x)
        if extension!='.rttm': continue
        with open(os.path.join(output_path, 'pred_rttms', x), 'r') as f:
            msg = [x.split() for x in f.readlines()]
        for l in msg:
            if l[0]=='SPEAKER':
                segs.append({'start': float(l[3]), 'end': float(l[3])+float(l[4]), 'label': l[7]})
        out.append({'filename': filename, 'segments': segs})
    return out

# Helper function to cleanup audios directory
def delete_files_in_directory_and_subdirectories(directory_path):
   try:
     for root, dirs, files in os.walk(directory_path):
       for file in files:
         file_path = os.path.join(root, file)
         os.remove(file_path)
     print("All files and subdirectories deleted successfully.")
   except OSError:
     print("Error occurred while deleting files and subdirectories.")


#
# Scoring (entry) script: entry point for execution, scoring script should contain two functions:
# * init(): this function should be used for any costly or common preparation for subsequent inferences, e.g.,
#           deserializing and loading the model into a global object.
# * run(mini_batch): The method to be parallelized. Each invocation will have one minibatch.
#       * mini_batch: Batch inference will invoke run method and pass either a list or Pandas DataFrame as an argument to the method.
#                     Each entry in min_batch will be - a filepath if input is a FileDataset, a Pandas DataFrame if input is a TabularDataset.
#       * return value: run() method should return a Pandas DataFrame or an array.
#                       For append_row output_action, these returned elements are appended into the common output file.
#                       For summary_only, the contents of the elements are ignored.
#                       For all output actions, each returned output element indicates one successful inference of input element in the input mini-batch.
#

def init():
    """Init"""
    # Managed output path to control where objects are returned
    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="ParallelRunStep Agent"
    )
    parser.add_argument("--event_type", type=str, default='telephonic')
    parser.add_argument("--max_num_speakers", type=int, default=3)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Folder structure
    Path('./input').mkdir(parents=True, exist_ok=True)
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path('./nemo_output').mkdir(parents=True, exist_ok=True)

    # Config files
    query_parameters = {"downloadformat": "yaml"}
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/components/diarize_inference/input/diar_infer_telephonic.yaml', params=query_parameters)
    with open("./input/diar_infer_telephonic.yaml", mode="wb") as f:
        f.write(response.content)
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/components/diarize_inference/input/diar_infer_meeting.yaml', params=query_parameters)
    with open("./input/diar_infer_meeting.yaml", mode="wb") as f:
        f.write(response.content)    

    # Diarization
    global msdd_model, msdd_cfg
    ## Read NeMo MSDD configuration file
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{args.event_type}.yaml')
    msdd_cfg.diarizer.clustering.max_num_speakers = args.max_num_speakers
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    ## Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=msdd_cfg)


def run(mini_batch):
    objs = []
    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        filename, extension = os.path.splitext(str(pathdir).split('/')[-1])
        if os.path.splitext(pathdir)[1] not in ['.wav', '.mp3']:
            log.info("Skipping file {}".format(pathdir))
            objs.append({'filename': pathdir, 'segments': []})
            continue
        log.info("Processing file {}".format(pathdir))

        # Preprocessing
        log.info(f"\tConvert to 16kHz mono")
        prep_time = time.time()
        signal, sample_rate = librosa.load(pathdir, sr=None) # load audio from storage
        if sample_rate!=16000: # Set sample_rate
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
        if len(signal.shape)>1: # Set num_channels
            signal = librosa.to_mono(signal)
        sf.write(f'./input_audios/{filename}{extension}', signal, 16000, 'PCM_24') # save in tmp path as 16kHz, mono
        prep_time = time.time() - prep_time
        log.info(f"\t\tPrep. time: {prep_time}")
    #
    # Speaker diarization
    #
    if len(os.listdir('./input_audios'))>0:
        # Read cfg
        log.info(f"Read NeMo MSDD configuration file:")
        filepaths = [os.path.join('./input_audios',x) for x in os.listdir('./input_audios')]
        create_msdd_config(filepaths) # initialise msdd cfg
        # Diarization
        log.info(f"Run diarization")
        diar_time = time.time()
        msdd_model.diarize()
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")
        # Process diarization output
        log.info(f"Save outputs")
        objs += process_NeMo_output('./nemo_output')
        delete_files_in_directory_and_subdirectories('./input_audios')
        delete_files_in_directory_and_subdirectories('./nemo_output')
    else:
        delete_files_in_directory_and_subdirectories('./input_audios')
        delete_files_in_directory_and_subdirectories('./nemo_output')

    return objs
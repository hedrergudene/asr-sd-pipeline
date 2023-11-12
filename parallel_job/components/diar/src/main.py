# Libraries
import argparse
import sys
import logging as log
import requests
from pathlib import Path
import os
import json
import time
from typing import List
import json
from typing import Dict
import torch
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from omegaconf import OmegaConf

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Helper function to build NeMo input manifes
def create_msdd_config(audio_filenames:List[str]):
    if os.path.exists("input/diar_manifest.jsonl"): os.remove("input/diar_manifest.jsonl")
    with open("input/diar_manifest.jsonl", "w") as fp:
        for x in audio_filenames:
            json.dump({
                    "audio_filepath": x,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "rttm_filepath": None,
                    "uem_filepath": None
               }, fp)
            fp.write('\n')


# Helper function to create voice activity detection manifest
def create_asr_vad_config(segments:Dict, filepath:str):
    fn, _ = os.path.splitext(filepath.split('/')[-1])
    asr_vad_manifest=[{"audio_filepath": filepath, "offset": float(x['start']), "duration": float(x['end'])-float(x['start']), "label": "UNK", "uniq_id": fn} for x in segments]
    if os.path.exists("./input/asr_vad_manifest.jsonl"): os.remove("./input/asr_vad_manifest.jsonl")
    with open("./input/asr_vad_manifest.jsonl", "w") as fp:
        for line in asr_vad_manifest:
            json.dump(line, fp)
            fp.write('\n')


# Helper function to process diarization output from method output
def process_diar_output(diar_output):
    return {fp:[{'start':float(x.split(' ')[0]), 'end': float(x.split(' ')[1]), 'speaker':x.split(' ')[2][-1]} for x in segments] for fp, segments in diar_output.items()}


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
    parser.add_argument("--input_audio_path", type=str)
    parser.add_argument("--event_type", type=str, default='telephonic')
    parser.add_argument("--max_num_speakers", type=int, default=3)
    parser.add_argument("--output_path", type=str)
    args, _ = parser.parse_known_args()

    # Diarization parameters
    global msdd_model, msdd_cfg, input_audio_path, output_path
    input_audio_path = args.input_audio_path
    output_path = args.output_path

    # Folder structure
    Path('./input').mkdir(parents=True, exist_ok=True)
    Path('./nemo_diar_output').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Config files
    query_parameters = {"downloadformat": "yaml"}
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/parallel_job/components/diar/src/input/diar_infer_telephonic.yaml', params=query_parameters)
    with open("./input/diar_infer_telephonic.yaml", mode="wb") as f:
        f.write(response.content)
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/parallel_job/components/diar/src/input/diar_infer_meeting.yaml', params=query_parameters)
    with open("./input/diar_infer_meeting.yaml", mode="wb") as f:
        f.write(response.content)

    # Read NeMo MSDD configuration file
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{args.event_type}.yaml')
    msdd_cfg.diarizer.clustering.parameters.max_num_speakers = args.max_num_speakers
    msdd_cfg.diarizer.vad.external_vad_manifest='./input/asr_vad_manifest.json'
    msdd_cfg.diarizer.asr.parameters.asr_based_vad = True
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    # Initialize NeMo MSDD diarization model
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)


def run(mini_batch):
    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        log.info(f"Processing file {fn}:")
        # Read word-level transcription to fetch timestamps
        with open(str(pathdir), 'r', encoding='utf-8') as f:
            x = json.load(f)['segments']
        # Ensure audio contains activity
        if len(x)==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"{output_path}/{fn}.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': [], # List of dictionaries with keys 'start', 'end'
                        'segments': []
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            continue
        word_ts = [[w['start'], w['end']] for segment in x for w in segment['words']]
        # Create ./input/asr_vad_manifest.json
        create_asr_vad_config(x, f'{input_audio_path}/{fn}.wav')

        #
        # Speaker diarization
        #
        log.info(f"Run diarization")
        diar_time = time.time()
        create_msdd_config([f"{input_audio_path}/{fn}.wav"]) # initialise msdd cfg
        msdd_model.audio_file_list = [f"{input_audio_path}/{fn}.wav"] # update audios list
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, {fn:word_ts})
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")
        # Process diarization output
        log.info(f"Save outputs")
        segments = process_diar_output(diar_hyp)[fn]
        with open(os.path.join(output_path, f"{fn}.json"), 'w', encoding='utf8') as f:
            json.dump(
                {
                    'segments': segments,
                    'metadata': {
                        'diarization_time': diar_time
                    }
                },
                f,
                indent=4,
                ensure_ascii=False
            )
        log.info(f"Cleanup resources")
        delete_files_in_directory_and_subdirectories('./nemo_diar_output')

    return mini_batch
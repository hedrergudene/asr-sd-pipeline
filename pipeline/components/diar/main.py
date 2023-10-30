# Libraries
import sys
import logging as log
from pathlib import Path
import os
import json
import time
from typing import List, Dict
import json
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from omegaconf import OmegaConf
import fire 

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


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    input_asr_path,
    event_type,
    max_num_speakers,
    output_path
):

    # Folder structure
    Path('./nemo_diar_output').mkdir(parents=True, exist_ok=True)

    ## Read NeMo MSDD configuration file
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{event_type}.yaml')
    msdd_cfg.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
    msdd_cfg.diarizer.asr.parameters.asr_based_vad = True
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    ## Initialize NeMo MSDD diarization model
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)
    # Set up input
    f = Path(input_path)
    files = list(f.iterdir())

    for pathdir in files:
        # Read file
        filename, _ = os.path.splitext(str(pathdir).split('/')[-1])
        log.info(f"Processing file {filename}:")
        # Read word-level transcription to fetch timestamps
        with open(os.path.join(input_asr_path, f"{filename}.json"), 'r', encoding='utf-8') as f:
            x = json.load(f)['segments']
        # Ensure audio contains activity
        if len(x)==0:
            log.info(f"Audio {filename} does not contain any activity. Generating dummy metadata:")
            with open(f"{output_path}/{filename}.json", 'w') as f:
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
        create_asr_vad_config(x, f'{input_path}/{filename}.wav')

        #
        # Speaker diarization
        #
        log.info(f"Run diarization")
        diar_time = time.time()
        create_msdd_config([f"{input_path}/{filename}.wav"]) # initialise msdd cfg
        msdd_model.audio_file_list = [f"{input_path}/{filename}.wav"] # update audios list
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, {filename:word_ts})
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")
        # Process diarization output
        log.info(f"Save outputs")
        segments = process_diar_output(diar_hyp)[filename]
        with open(os.path.join(output_path, f"{filename}.json"), 'w', encoding='utf8') as f:
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


if __name__=="__main__":
    fire.Fire(main)
# Libraries
import sys
import subprocess as sp
import logging as log
from pathlib import Path
import os
import json
import time
from typing import Dict
import json
import fire 

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Helper function to prepare ASR output to be aligned. Segments separators are '|'
def create_nfa_config(segments:Dict, filepath:str):
    if os.path.exists("input/nfa_manifest.jsonl"): os.remove("input/nfa_manifest.jsonl")
    with open("input/nfa_manifest.jsonl", "w") as fp:
        json.dump({
            "audio_filepath": filepath,
            "text": ' | '.join([x['text'] for x in segments])
        }, fp)
        fp.write('\n')

# Helper function to process forced alignment output
def process_nfa_output(filename):
    # Get word-level timestamps
    with open(f"./nemo_nfa_output/ctm/segments/{filename}.ctm", 'r') as f:
        sentence_level_ts = f.read().split('\n')[:-1]
        sentence_level_ts = [{'start':float(y.split(' ')[2]), 'end':float(y.split(' ')[2])+float(y.split(' ')[3]), 'text':y.split(' ')[-1].replace('<space>', ' ')} for y in sentence_level_ts]
    with open(f"./nemo_nfa_output/ctm/words/{filename}.ctm", 'r') as f:
        word_level_ts = f.read().split('\n')[:-1]
        word_level_ts = [{'start':float(y.split(' ')[2]), 'end':float(y.split(' ')[2])+float(y.split(' ')[3]), 'text':y.split(' ')[-1]} for y in word_level_ts]
    sg = []
    shift=0
    for h in sentence_level_ts:
        sg.append(
            {
                'start':h['start'],
                'end':h['end'],
                'text':h['text'],
                'words':word_level_ts[shift:shift+len(h['text'].split(' '))]
            }
        )
        shift+=len(h['text'].split(' '))
    return sg

# Helper function to cleanup audios directory
def delete_files_in_directory_and_subdirectories(directory_path):
   try:
     for root, dirs, files in os.walk(directory_path):
       for file in files:
         file_path = os.path.join(root, file)
         os.remove(file_path)
     log.info("All files and subdirectories deleted successfully.")
   except OSError:
     log.info("Error occurred while deleting files and subdirectories.")


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    input_asr_path,
    fa_model_name,
    fa_batch_size,
    output_path
):

    # Folder structure
    Path('./NeMo').mkdir(parents=True, exist_ok=True)
    Path('./nemo_nfa_output').mkdir(parents=True, exist_ok=True)

    # Set up input
    f = Path(input_asr_path)
    files = list(f.iterdir())

    # Clone repo
    result = sp.run(
        [
            'git',
            'clone',
            'https://github.com/NVIDIA/NeMo',
            '-b',
            'v1.20.0',
            './NeMo'
        ],
        capture_output=True,
        text=True
    )
    # Check return code
    if result.returncode!=0:
        log.error(f"NeMo repo cloning raised an exception: {result.stderr}")
        raise RuntimeError(f"NeMo repo cloning raised an exception: {result.stderr}")

    for pathdir in files:
        # Read file
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        log.info(f"Processing file {fn}:")
    
        # Read word-level transcription to fetch timestamps
        with open(os.path.join(input_asr_path, f"{fn}.json"), 'r', encoding='utf-8') as f:
            asr_dct = json.load(f)
        # Ensure audio contains activity
        if len(asr_dct['segments'])==0:
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
        # Create config
        create_nfa_config(asr_dct['segments'], f'{input_path}/{fn}.wav')

        #
        # Forced alignment
        #
        log.info(f"Run alignment")
        # Run script
        align_time = time.time()
        result = sp.run(
            [
                sys.executable,
                'NeMo/tools/nemo_forced_aligner/align.py',
                f'pretrained_name="{fa_model_name}"',
                'manifest_filepath="./input/nfa_manifest.jsonl"',
                'output_dir="./nemo_nfa_output"',
                f'batch_size={fa_batch_size}',
                'additional_segment_grouping_separator="|"'
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='xmlcharrefreplace'
        )
        align_time = time.time() - align_time
        # Check return code
        if ((result.returncode!=0) & (asr_dct['segments'][0].get('words') is None)):
            log.error(f"Alignment raised an exception and there are no timestamps available from ASR: {result.stderr}")
            raise RuntimeError(f"Alignment raised an exception and there are no timestamps available from ASR: {result.stderr}")
        elif ((result.returncode!=0) & (asr_dct['segments'][0].get('words') is not None)):
            log.warning(f"Alignment raised an exception; using ASR word-level timestamps: {result.stderr}")
            # Process output
            with open(os.path.join(output_path, f"{fn}.json"), 'w', encoding='utf8') as f:
                json.dump(
                    {
                        'segments': asr_dct['segments'],
                        'duration': asr_dct['duration'],
                        'vad_timestamps': asr_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                        'metadata': {**asr_dct['metadata'], **{'alignment_time': align_time}}
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
        elif ((result.returncode==0) & (asr_dct['segments'][0].get('words') is None)):
            log.info(f"Alignment run successfully. Including word-level timestamps.")
            # Update timestamps from both segment-level and word-level information
            segments = process_nfa_output(fn)
            # Process output
            with open(os.path.join(output_path, f"{fn}.json"), 'w', encoding='utf8') as f:
                json.dump(
                    {
                        'segments': segments,
                        'duration': asr_dct['duration'],
                        'vad_timestamps': asr_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                        'metadata': {**asr_dct['metadata'], **{'alignment_time': align_time}}
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
        else:
            # Update timestamps from both segment-level and word-level information
            log.info(f"Alignment run successfully. Updating word-level timestamps.")
            segments = process_nfa_output(fn)

            # Keep confidence results from ASR
            for asr_seg, nfa_seg in zip(asr_dct['segments'], segments):
                for asr_word, nfa_word in zip(asr_seg['words'], nfa_seg['words']):
                    nfa_word['confidence'] = asr_word['confidence']

            # Process output
            with open(os.path.join(output_path, f"{fn}.json"), 'w', encoding='utf8') as f:
                json.dump(
                    {
                        'segments': segments,
                        'duration': asr_dct['duration'],
                        'vad_timestamps': asr_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                        'metadata': {**asr_dct['metadata'], **{'alignment_time': align_time}}
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
        log.info(f"Cleanup resources")
        delete_files_in_directory_and_subdirectories('./nemo_nfa_output')


if __name__=="__main__":
    fire.Fire(main)
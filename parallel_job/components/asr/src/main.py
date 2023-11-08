# Libraries
import argparse
import sys
import logging as log
from pathlib import Path
import os
import time
import json
import torch
import pandas as pd
from faster_whisper import WhisperModel

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

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
    parser.add_argument("--model_name", type=str, default='large-v2')
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--word_level_timestamps", type=bool, default=True)
    parser.add_argument("--condition_on_previous_text", type=bool, default=True)
    parser.add_argument("--compute_type", type=str, default='float16')
    parser.add_argument("--language_code", type=str, default='es')
    parser.add_argument("--output_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    global input_audio_path, beam_size, word_level_timestamps, condition_on_previous_text, language_code, output_path
    input_audio_path = args.input_audio_path
    beam_size = args.beam_size
    word_level_timestamps = args.word_level_timestamps
    condition_on_previous_text = args.condition_on_previous_text
    language_code = args.language_code
    output_path = args.output_path

    # Folder structure
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ASR models
    global whisper_model
    whisper_model = WhisperModel(
        args.whisper_model_name,
        device=device,
        compute_type=args.compute_type
    )


def run(mini_batch):
    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        fn, ext = os.path.splitext(str(pathdir).split('/')[-1])

        # Fetch metadata
        log.info(f"Processing file {fn}:")
        with open(str(pathdir), 'r') as f:
            metadata_dct = json.load(f)

        # Ensure audio contains activity
        if len(metadata_dct['vad_timestamps'])==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"{output_path}/{fn}.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': metadata_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                        'segments': []
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            continue

        #
        # Transcription
        #
        log.info(f"\tASR:")
        transcription_time = time.time()
        segments, _ = whisper_model.transcribe(
            f"{input_audio_path}/{fn}{ext}",
            beam_size=beam_size,
            language=language_code,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=False,
            word_timestamps=word_level_timestamps
        )

        if word_level_timestamps:
            segs = []
            for x in segments:
                words = []
                for word in x.words:
                    words.append(
                       {
                          'start':word.start,
                          'end':word.end,
                          'text':word.word.strip(),
                          'confidence': word.probability
                       }
                    )
                s = {
                   'start':words[0]['start'],
                   'end':words[-1]['end'],
                   'text':' '.join([w['text'] for w in words]),
                   'confidence': sum([w['confidence'] for w in words])/len([w['confidence'] for w in words])
                }
                s['words'] = words
                segs.append(s)
        else:
            segs = [{'start': x.start, 'end': x.end, 'text': x.text.strip()} for x in segments]
        transcription_time = time.time()-transcription_time
        log.info(f"\t\tTranscription time: {transcription_time}")

        # Build metadata
        mtd = {
            "transcription_time": transcription_time
        }
        # Save output
        with open(os.path.join(output_path, f"{fn}.json"), 'w', encoding='utf8') as f:
            json.dump(
                {
                    'segments': segs,
                    'duration': metadata_dct['duration'],
                    'vad_timestamps': metadata_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                    'metadata': {**metadata_dct['metadata'], **mtd}
                },
                f,
                indent=4,
                ensure_ascii=False
            )

    return mini_batch
# Libraries
import argparse
import sys
import logging as log
import soundfile as sf
import librosa
from pathlib import Path
import os
import re
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
    parser.add_argument("--whisper_model_name", type=str, default='large-v2')
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--vad_threshold", type=float, default=0.5)
    parser.add_argument("--min_speech_duration_ms", type=int, default=200)
    parser.add_argument("--min_silence_duration_ms", type=int, default=500)
    parser.add_argument("--compute_type", type=str, default='float16')
    parser.add_argument("--language_code", type=str, default='es')
    parser.add_argument("--output_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    global beam_size, vad_threshold, min_speech_duration_ms, min_silence_duration_ms, language_code, output_path
    beam_size = args.beam_size
    vad_threshold = args.vad_threshold
    min_speech_duration_ms = args.min_speech_duration_ms
    min_silence_duration_ms = args.min_silence_duration_ms
    language_code = args.language_code
    output_path = args.output_path

    # Folder structure
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
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
        filename, extension = os.path.splitext(str(pathdir).split('/')[-1])
        if os.path.splitext(pathdir)[1] not in ['.wav', '.mp3']:
            log.info("Skipping file {}".format(pathdir))
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
        sf.write(f'./input_audios/{filename}.wav', signal, 16000, 'PCM_24') # save in tmp path as 16kHz, mono
        prep_time = time.time() - prep_time
        log.info(f"\t\tPrep. time: {prep_time}")

        #
        # Transcription
        #
        log.info(f"\tASR:")
        transcription_time = time.time()
        segments, _ = whisper_model.transcribe(
            signal,
            beam_size=beam_size,
            language=language_code,
            vad_filter=True,
            word_timestamps=False,
            vad_parameters=dict(
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        )
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
        transcription_time = time.time()-transcription_time
        log.info(f"\t\tTranscription time: {transcription_time}")

        # Build metadata
        mtd = {
            "preprocessing_time": prep_time,
            "transcription_time": transcription_time
        }
        # Save output
        with open(os.path.join(output_path, f"{filename}.json"), 'w', encoding='utf8') as f:
            json.dump(
                {
                    'segments': segs,
                    'duration': librosa.get_duration(y=signal, sr=16000),
                    'metadata': mtd
                },
                f,
                indent=4,
                ensure_ascii=False
            )
        ## Generate output (filename goes WITHOUT extension, we no longer give a f**k!)
        #objs.append({'filename': filename, 'segments': aligned_output})       

    # Remove audios
    delete_files_in_directory_and_subdirectories('./input_audios')

    return mini_batch
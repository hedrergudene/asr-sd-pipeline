# Requierments
##Essentials
import logging as log
import subprocess
import json
import os
import sys
from pathlib import Path
import time
import shlex
import numpy as np
import argparse
## Audio processing
import demucs.separate
import torch

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Helper method to decode an audio
def preprocess_audio(input_path, output_path, filename):
    """Method to preprocess audios with ffmpeg, using the following configuration:
        * '-acodec': Specifies the audio codec to be used. In this case, it's set to 'pcm_s16le',
                     which stands for 16-bit little-endian PCM (Pulse Code Modulation).
                     This is a standard audio format.
        * '-ac' '1': Sets the number of audio channels to 1, which is mono audio.
        * '-ar' '16000': Sets the audio sample rate to 16 kHz.

    Args:
        input_filepath (str): Folder where audio lies
        output_filepath (str): Folder where audio is to be stored after processing
        filename (str): Name of the file (with extension) you are processing.
    """
    fn, ext = os.path.splitext(filename)
    command = ['ffmpeg', '-i', f"{input_path}/{filename}", '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', f"{output_path}/{fn}.wav"]
    out = subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    if out.returncode!=0:
        raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")

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
    parser.add_argument("--vad_threshold", type=float, default=0.75)
    parser.add_argument("--min_speech_duration_ms", type=int, default=250)
    parser.add_argument("--min_silence_duration_ms", type=int, default=500)
    parser.add_argument("--use_onnx_vad", type=bool, default=False)
    parser.add_argument("--demucs_model", type=str, default='htdemucs')
    parser.add_argument("--output_audios_path", type=str)
    parser.add_argument("--output_metadata_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    global vad_threshold, min_speech_duration_ms, min_silence_duration_ms, use_onnx_vad, demucs_model, output_audios_path, output_metadata_path
    vad_threshold = args.vad_threshold
    min_speech_duration_ms = args.min_speech_duration_ms
    min_silence_duration_ms = args.min_silence_duration_ms
    use_onnx_vad = args.use_onnx_vad
    demucs_model = args.demucs_model
    output_audios_path = args.output_audios_path
    output_metadata_path = args.output_metadata_path

    # Folder structure
    Path('./prep_audios').mkdir(parents=True, exist_ok=True)
    Path(output_audios_path).mkdir(parents=True, exist_ok=True)
    Path(output_metadata_path).mkdir(parents=True, exist_ok=True)

    # VAD model
    global vad_model, get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=use_onnx_vad)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

def run(mini_batch):
    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        fn, ext = os.path.splitext(str(pathdir).split('/')[-1])
        if ext not in ['.wav', '.mp3']:
            log.info("Skipping file {}".format(fn))
            continue
        log.info("Processing file {}:".format(fn))
        prep_time = time.time()
        
        # Standarise format
        preprocess_audio('/'.join(str(pathdir).split('/')[:-1]), './prep_audios', fn)

        # VAD
        wav = read_audio(f"./prep_audios/{fn}", sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            threshold=vad_threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )
        if len(speech_timestamps)==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"{output_metadata_path}/{fn}.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': speech_timestamps, # List of dictionaries with keys 'start', 'end'
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            continue
        save_audio(f"./prep_audios/{fn}_vad{ext}",
                 collect_chunks(speech_timestamps, wav), sampling_rate=16000)
        audio_length_s = len(wav)/16000
        vad_length_s = sum([(s['end']-s['start']) for s in speech_timestamps])/16000
        log.info(f"\tVAD filtered {np.round((vad_length_s/audio_length_s)*100,2)}% of audio. Remaining audio length: {np.round(vad_length_s,2)}s")

        # Demucs
        demucs.separate.main(shlex.split(f'--two-stems vocals -o "./prep_audios" -n {demucs_model} "./prep_audios/{fn}_vad{ext}"'))

        # Convert demucs output to mono signal
        command = [
            'ffmpeg',
            '-i',
            f"./prep_audios/{demucs_model}/{fn}_vad/vocals.wav",
            '-ac',
            '1',
            '-ar',
            '16000',
            f"{output_audios_path}/{fn}.wav"
        ]
        out = subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        if out.returncode!=0:
            raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")
        prep_time = time.time() - prep_time
        log.info(f"\tRutime: {prep_time}")
        
        # Write metadata file
        with open(f"{output_metadata_path}/{fn}.json", 'w') as f:
            json.dump(
                {
                    'vad_timestamps': speech_timestamps, # List of dictionaries with keys 'start', 'end'
                    'duration': audio_length_s,
                    'metadata': {
                       'preprocessing_time': prep_time
                    }
                },
                f,
                indent=4,
                ensure_ascii=False
            )
        
        # Cleanup resources
        delete_files_in_directory_and_subdirectories('./prep_audios')

    return mini_batch
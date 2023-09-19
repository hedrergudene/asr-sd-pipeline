# Libraries
import sys
import logging as log
import soundfile as sf
import librosa
from pathlib import Path
import os
import json
import time
from typing import List
import json
import torch
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
                    "uem_filepath": None
               }, fp)
            fp.write('\n')

# Helper function to create voice activity detection manifest
def create_asr_vad_config(segments, filepath, filename):
    return [{"audio_filepath": filepath, "offset": float(x['start']), "duration": float(x['end'])-float(x['start']), "label": "UNK", "uniq_id": filename} for x in segments]


# Helper function to process diarization output from method output
def process_NeMo_output(diar_output):
    return [{'filename': fp, 'segments':[{'start':float(x.split(' ')[0]), 'end': float(x.split(' ')[1]), 'speaker':x.split(' ')[2][-1]} for x in segments]} for fp, segments in diar_output.items()]


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

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Folder structure
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path('./nemo_output').mkdir(parents=True, exist_ok=True)

    ## Read NeMo MSDD configuration file
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{event_type}.yaml')
    msdd_cfg.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
    msdd_cfg.diarizer.asr.parameters.asr_based_vad = True
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    ## Initialize NeMo MSDD diarization model
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)


    word_ts = {}
    asr_vad_manifest = []

    # Set up input
    f = Path(input_path)
    files = list(f.iterdir())

    for pathdir in files:
        # Read file
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

        # Read word-level transcription to fetch timestamps
        with open(os.path.join(input_asr_path, f"{filename}.json"), 'r', encoding='utf-8') as f:
            x = json.load(f)['segments']
        word_ts[filename] = [[w['start'],w['end']] for s in x for w in s['words']]
        # Fetch VAD info
        asr_vad_manifest += create_asr_vad_config(x, f'./input_audios/{filename}.wav', filename)
    # Create ./nemo_output/asr_vad_manifest.json
    if os.path.exists("./nemo_output/asr_vad_manifest.json"): os.remove("./nemo_output/asr_vad_manifest.json")
    with open("./nemo_output/asr_vad_manifest.json", "w") as fp:
        for line in asr_vad_manifest:
            json.dump(line, fp)
            fp.write('\n')

    #
    # Speaker diarization
    #
    log.info(f"Read NeMo MSDD configuration file:")
    filepaths = [os.path.join('./input_audios',x) for x in os.listdir('./input_audios')]

    # Diarization
    log.info(f"Run diarization")
    for f in filepaths:
        diar_time = time.time()
        create_msdd_config([f]) # initialise msdd cfg
        msdd_model.audio_file_list = [f] # update audios list
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, word_ts)
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")
        # Process diarization output
        log.info(f"Save outputs")
        for x in process_NeMo_output(diar_hyp):
            with open(os.path.join(output_path, f"{x['filename']}.json"), 'w', encoding='utf8') as f:
                json.dump(
                    {
                        'segments': x['segments'],
                        'metadata': {
                            'diarization_time': diar_time
                        }
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            os.remove(f"./input_audios/{x['filename']}.wav")
    log.info(f"Cleanup resources")
    delete_files_in_directory_and_subdirectories('./nemo_output')


if __name__=="__main__":
    fire.Fire(main)
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
import fire
## KeyVaults
from cryptography.fernet import Fernet
## Audio processing
import demucs.separate
import torch
## Azure
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

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
    

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    keyvault_name,
    secret_name,
    vad_threshold,
    min_speech_duration_ms,
    min_silence_duration_ms,
    use_onnx_vad,
    demucs_model,
    output_audios_path,
    output_metadata_path
):
    # Create output paths
    Path('./decrypted_audios').mkdir(parents=True, exist_ok=True)
    Path('./prep_audios').mkdir(parents=True, exist_ok=True)
    Path(output_audios_path).mkdir(parents=True, exist_ok=True)
    Path(output_metadata_path).mkdir(parents=True, exist_ok=True)

    # Set up keyvault client
    log.info("Setting up keyvault client:")
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    secret_client = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
    # Fetch secret
    key = secret_client.get_secret(secret_name).value
    if isinstance(key, str):
        key = key.encode()
    # Pick up KeyVault
    my_fernet = Fernet(key)

    # VAD model
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=use_onnx_vad)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    for filename in os.listdir(input_path):
        fn, ext = filename
        if ext not in ['.wav', '.mp3']:
            log.info("Skipping file {}".format(fn))
            continue
        log.info("Processing file {}:".format(fn))
        prep_time = time.time()
        
        # Decrypt file
        with open(f"{input_path}/{filename}", 'rb') as f:
            encrypted_bytes = f.read()
        decrypted_bytes = my_fernet.encrypt(encrypted_bytes)
        # Save decrypted data
        with open(f'./decrypted_audios/{filename}', 'wb') as fd:
            fd.write(decrypted_bytes)
        
        # Standarise format
        preprocess_audio('./decrypted_audios', './prep_audios', filename)

        # VAD
        wav = read_audio(f"./prep_audios/{filename}", sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            threshold=vad_threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )
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

        ## Encrypt file and store in output path
        #with open(f"./prep_audios/{fn}_vad_demucs_mono.wav", 'rb') as f:
        #    decrypted_bytes = f.read()
        #encrypted_bytes = my_fernet.encrypt(decrypted_bytes)
        ## Save decrypted data
        #with open(f'{output_audios_path}/{filename}', 'wb') as fd:
        #    fd.write(encrypted_bytes)
        
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
        delete_files_in_directory_and_subdirectories('./decrypted_audios')
        delete_files_in_directory_and_subdirectories('./prep_audios')

if __name__=="__main__":
    fire.Fire(main)
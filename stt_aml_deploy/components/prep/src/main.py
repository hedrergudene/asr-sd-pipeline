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
from typing import List, Dict, Tuple, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import pgpy
import pymongo
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

# Keyvault handling class
class CredentialManager():
    def __init__(
            self,
            keyvault_name:str,
            secret_tenant_sp:str=None,
            secret_client_sp:str=None,
            secret_sp:str=None,
            puk_secret_name:str=None,
            prk_secret_name:str=None,
            prk_password_secret_name:str=None,
    ) -> None:
        """Base class to handle PGP encryption system in Azure.

        Args:
            keyvault_name (str): KeyVault resource where secrets are stored.
            secret_tenant_sp (str): Service principal tenant_id secret, stored in KeyVault 'keyvault_name'.
            secret_client_sp (str): Service principal client_id secret, stored in KeyVault 'keyvault_name'.
            secret_sp (str): Service principal secret_client secret, stored in KeyVault 'keyvault_name'.
        """
        self.keyvault_name = keyvault_name
        self.secret_tenant_sp = secret_tenant_sp
        self.secret_client_sp = secret_client_sp
        self.secret_sp = secret_sp
        self.login = None
        # Import public key from PGP
        puk_secret_value = self.fetch_secret(self.default_login(), puk_secret_name)
        self.public_key, _ = pgpy.PGPKey.from_blob(puk_secret_value)
        # Retrieve pk secrets
        sc = self.sp_login()
        self.enable_secret(sc, prk_secret_name, True)
        self.enable_secret(sc, prk_password_secret_name, True)
        pk_secret_value = self.fetch_secret(sc, prk_secret_name)
        self.pk_pass_secret_value = self.fetch_secret(sc, prk_password_secret_name)
        # Fetch pk key
        self.private_key, _ = pgpy.PGPKey.from_blob(pk_secret_value)


    def encrypt(
            self,
            input_path:str,
            output_path:str,
            filenames:List[str],
            remove_input:bool=False,
            secret_client:SecretClient=None
    ) -> None:
        # Check input is a list
        if isinstance(filenames, str):
            filenames = [filenames]
        # Default login
        if ((self.login!='default') | (secret_client is None)):
            secret_client = self.default_login()
        # Loop
        for filename in filenames:
            input_filepath = os.path.join(input_path, filename)
            output_filepath = os.path.join(output_path, filename)
            folder_path, fn, ext = self.get_file_attr(input_filepath)
            if ext=='.pgp':
                log.warning(f"File {fn} is already encrypted. Skipping...")
                continue
            with open(input_filepath, 'rb') as f:
                message = pgpy.PGPMessage.new(f.read())
            encrypted_message = self.public_key.encrypt(message)
            encrypted_message = str(encrypted_message)
            with open(output_filepath+'.pgp', "w") as f:
                f.write(encrypted_message)
            log.info(f"File {fn+ext+'.pgp'} has been generated in {folder_path}.")
            if remove_input: os.remove(input_filepath)
            log.info(f"File {fn+ext} has been removed.")


    def decrypt(
            self,
            input_path: str,
            output_path: str,
            filenames:List[str],
            remove_input:bool=False,
            secret_client:SecretClient=None
    ) -> None:
        # Check input is a list
        if isinstance(filenames, str):
            filenames = [filenames]
        # Service principal login
        if ((self.login!='sp') | (secret_client is None)):
            secret_client = self.sp_login()
        # Loop
        for filename in filenames:
            input_filepath = os.path.join(input_path, filename)
            folder_path, fn, ext = self.get_file_attr(input_filepath)
            if ext not in ['.pgp', '.enc']:
                log.warning(f"File {fn} is already decrypted. Skipping...")
                continue
            with self.private_key.unlock(self.pk_pass_secret_value) as ukey:
                if ukey:
                    encrypted_message = pgpy.PGPMessage.from_file(input_filepath)
                    decrypted_message = ukey.decrypt(encrypted_message).message
                    if isinstance(decrypted_message, str):
                        with open(os.path.join(output_path, fn), "w") as f:
                            f.write(decrypted_message)
                    elif isinstance(decrypted_message, bytearray):
                        with open(os.path.join(output_path, fn), "wb") as f:
                            f.write(decrypted_message)
                    else:
                        log.error(f"File {fn} returned a decrypted message that it's not either str not bytearray. Please check.")
                        raise ValueError(f"File {fn} returned a decrypted message that it's not either str not bytearray. Please check.")
                    log.info(f"File {fn} has been generated in {folder_path}.")
                    if remove_input: os.remove(input_filepath)
                    log.info(f"File {fn+ext} has been removed.")
                else:
                    log.error(f"Private key password is not correct.")
                    raise ValueError(f"Private key password is not correct.")


    def default_login(self) -> SecretClient:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        secret_client = SecretClient(vault_url=f"https://{self.keyvault_name}.vault.azure.net/", credential=credential)
        self.login = 'default'
        return secret_client


    def sp_login(self) -> SecretClient:
        # Make sure all parameters are in place
        if ((self.secret_tenant_sp is None) | (self.secret_client_sp is None) | (self.secret_sp is None)):
            log.error(f"Service principal credentials have not been set up properly.")
            raise ValueError(f"Service principal credentials have not been set up properly.")
        # Get secret client
        secret_client = self.default_login()
        tenant_id = secret_client.get_secret(name=self.secret_tenant_sp).value
        client_id = secret_client.get_secret(name=self.secret_client_sp).value
        client_secret = secret_client.get_secret(name=self.secret_sp).value
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        secret_client = SecretClient(vault_url=f"https://{self.keyvault_name}.vault.azure.net/", credential=credential)
        self.login = 'sp'
        return secret_client


    def enable_secret(
            self,
            secret_client:SecretClient=None,
            secret_name:str=None,
            enable:bool=False
    ) -> None:
        # Get the right login for the operation
        if self.login!='sp':
            secret_client = self.sp_login()
        # Check secret current status
        try:
            secret_status = secret_client.get_secret(secret_name).properties.enabled
        except:
            secret_status = False
        # Compare with input action
        if secret_status==enable:
            s = 'enabled' if enable else 'disabled'
            log.info(f"Secret {secret_name} is already {s}.")
        else:
            s = 'enabled' if enable else 'disabled'
            secret_client.update_secret_properties(secret_name, enabled=enable)
            log.info(f"Secret {secret_name} is now {s}.")


    def fetch_secret(
            self,
            secret_client:SecretClient,
            secret_name:str
    ) -> str:
        secret_value = secret_client.get_secret(secret_name).value
        return secret_value


    @staticmethod
    def get_file_attr(
            filepath:str
    ) -> List[str]:
        """Helper function to consistently split a filepath into folder path, filename and extension.

        Args:
            filepath (str): Path where file is stored.

        Returns:
            List[str]: Folder path, file name and file extension.
        """
        folder_path = '/'.join(filepath.split('/')[:-1])
        fn, ext = os.path.splitext(filepath.split('/')[-1])
        return folder_path, fn, ext


# Helper method to decode an audio
def preprocess_audio(input_filepath, output_filepath, filename):
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
    command = ['ffmpeg', '-i', f"{input_filepath}/{filename}", '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', f"{output_filepath}/{fn}.wav"]
    out = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    if out.returncode!=0:
        raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")


# Helper method to cleanup audios directory
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
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--secret_tenant_sp", type=str)
    parser.add_argument("--secret_client_sp", type=str)
    parser.add_argument("--secret_sp", type=str)
    parser.add_argument("--pk_secret", type=str)
    parser.add_argument("--pk_pass_secret", type=str)
    parser.add_argument("--pubk_secret", type=str)
    parser.add_argument("--cosmosdb_name", type=str)
    parser.add_argument("--cosmosdb_collection", type=str)
    parser.add_argument("--cosmosdb_cs_secret", type=str)
    parser.add_argument("--vad_threshold", type=float, default=0.75)
    parser.add_argument("--min_speech_duration_ms", type=int, default=250)
    parser.add_argument("--min_silence_duration_ms", type=int, default=500)
    parser.add_argument("--demucs_model", type=str, default='htdemucs')
    parser.add_argument("--output_prep_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Encrypt params
    global keyvault_name, secret_tenant_sp, secret_client_sp, secret_sp, pk_secret, pk_pass_secret, pubk_secret
    keyvault_name = args.keyvault_name
    secret_tenant_sp = args.secret_tenant_sp
    secret_client_sp = args.secret_client_sp
    secret_sp = args.secret_sp
    pk_secret = args.pk_secret
    pk_pass_secret = args.pk_pass_secret
    pubk_secret = args.pubk_secret


    # Preprocess params
    global vad_threshold, min_speech_duration_ms, min_silence_duration_ms, demucs_model, output_prep_path
    vad_threshold = args.vad_threshold
    min_speech_duration_ms = args.min_speech_duration_ms
    min_silence_duration_ms = args.min_silence_duration_ms
    demucs_model = args.demucs_model
    output_prep_path = args.output_prep_path

    # Folder structure
    Path('./decrypted_files').mkdir(parents=True, exist_ok=True)
    Path('./prep_audios').mkdir(parents=True, exist_ok=True)
    Path('./trimmed_audios').mkdir(parents=True, exist_ok=True)
    Path(output_prep_path).mkdir(parents=True, exist_ok=True)

    # VAD model
    global vad_model, get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True
                                  )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Instantiate credential manager
    global cm
    cm = CredentialManager(keyvault_name, secret_tenant_sp, secret_client_sp, secret_sp, pubk_secret, pk_secret, pk_pass_secret)

    # MongoDB client
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    sc = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
    connection_string = sc.get_secret(name=args.cosmosdb_cs_secret).value
    mongodb_client = pymongo.MongoClient(connection_string)
    # DB connection
    if args.cosmosdb_name not in mongodb_client.list_database_names():
        log.error(f"Database {args.cosmosdb_name} not found.")
        raise ValueError(f"Database {args.cosmosdb_name} not found.")
    else:
        cosmosdb_db = mongodb_client[args.cosmosdb_name]
        log.info(f"Database {args.cosmosdb_name} connected.")
    # Collection connection
    if args.cosmosdb_collection not in cosmosdb_db.list_collection_names():
        log.error(f"Collection {args.cosmosdb_collection} not found.")
        raise ValueError(f"Collection {args.cosmosdb_collection} not found.")
    else:
        global cosmosdb_client
        cosmosdb_client = cosmosdb_db[args.cosmosdb_collection]
        log.info(f"Collection {args.cosmosdb_collection} connected.")


def run(mini_batch):

    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        input_folder = '/'.join(str(pathdir).split('/')[:-1])
        fn, ext_enc = os.path.splitext(str(pathdir).split('/')[-1])
        fn, ext_file = os.path.splitext(fn)
        if ext_enc not in ['.pgp', '.enc']:
            if ext_enc in ['.wav', '.mp3']:
                log.warning(f"Processing unencrypted file {fn}.")
            else:
                log.info(f"Skipping file {fn}, encoding extension not valid ('{ext_enc}')")
                continue
        elif ext_file not in ['.wav', '.mp3']:
            log.info(f"Skipping file {fn}, file extension not valid ('{ext_file}')")
            continue
        elif cosmosdb_client.find_one({"_id": fn}) is not None:
            log.info(f"Skipping file {fn}, record already found in cosmosDB collection.")
            continue            
        else:
            log.info(f"Processing file {fn}:")
        prep_time = time.time()

        # Standarise format
        if not os.path.isfile(str(pathdir)): raise ValueError(f"Filepath does not exist.")
        if ext_enc in ['.pgp', '.enc']:
            log.info(f"Decrypting and preprocessing:")
            cm.decrypt(input_folder, './decrypted_files', [f"{fn}{ext_file}{ext_enc}"])
            preprocess_audio('./decrypted_files', './prep_audios', f"{fn}{ext_file}")
        else:
            log.info(f"Preprocessing:")
            ext_file = ext_enc
            preprocess_audio(input_folder, './prep_audios', f"{fn}{ext_file}")

        # VAD
        log.info(f"Get speech activity timestamps:")
        wav = read_audio(f"./prep_audios/{fn}.wav", sampling_rate=16000)
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
            with open(f"./decrypted_files/{fn}_prep.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': speech_timestamps, # List of dictionaries with keys 'start', 'end'
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            cm.encrypt('./decrypted_files', output_prep_path, [f"{fn}_prep.json"], True)
            continue
        save_audio(f"./prep_audios/{fn}_vad.wav",
                 collect_chunks(speech_timestamps, wav), sampling_rate=16000)
        audio_length_s = len(wav)/16000
        vad_length_s = sum([(s['end']-s['start']) for s in speech_timestamps])/16000
        log.info(f"\tVAD filtered {np.round((vad_length_s/audio_length_s)*100,2)}% of audio. Remaining audio length: {np.round(vad_length_s,2)}s")

        # Demucs
        log.info(f"Apply demucs:")
        demucs.separate.main(shlex.split(f'--two-stems vocals -o "./prep_audios" -n {demucs_model} "./prep_audios/{fn}_vad.wav"'))

        # Convert demucs output to mono signal
        log.info(f"Standardise demucs output:")
        command = [
            'ffmpeg',
            '-i',
            f"./prep_audios/{demucs_model}/{fn}_vad/vocals.wav",
            '-ac',
            '1',
            '-ar',
            '16000',
            f"./trimmed_audios/{fn}.wav"
        ]
        out = subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        if out.returncode!=0:
            raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")
        
        # Encrypt output
        log.info(f"Encrypt audio output:")
        cm.encrypt('./trimmed_audios', output_prep_path, [f"{fn}.wav"], True)

        prep_time = time.time() - prep_time
        log.info(f"\tRutime: {prep_time}")
        
        # Write metadata file
        with open(f"./decrypted_files/{fn}_prep.json", 'w') as f:
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
        cm.encrypt('./decrypted_files', output_prep_path, [f"{fn}_prep.json"], True)
        
        # Cleanup resources
        delete_files_in_directory_and_subdirectories('./decrypted_files')
        delete_files_in_directory_and_subdirectories('./prep_audios')
        delete_files_in_directory_and_subdirectories('./trimmed_audios')

    return mini_batch


def shutdown():
    cm.enable_secret(cm.sp_login(), pk_secret, False)
    cm.enable_secret(cm.sp_login(), pk_pass_secret, False)
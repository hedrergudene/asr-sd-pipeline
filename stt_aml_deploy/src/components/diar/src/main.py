# Libraries
import argparse
import sys
import logging as log
import requests
from pathlib import Path
import os
import re
import json
import time
from typing import List, Dict, Tuple, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import pgpy
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
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--secret_tenant_sp", type=str)
    parser.add_argument("--secret_client_sp", type=str)
    parser.add_argument("--secret_sp", type=str)
    parser.add_argument("--pk_secret", type=str)
    parser.add_argument("--pk_pass_secret", type=str)
    parser.add_argument("--pubk_secret", type=str)
    parser.add_argument("--event_type", type=str, default='telephonic')
    parser.add_argument("--max_num_speakers", type=int, default=3)
    parser.add_argument("--min_window_length", type=float, default=0.2)
    parser.add_argument("--overlap_threshold", type=float, default=0.8)
    parser.add_argument("--output_diar_path", type=str)
    args, _ = parser.parse_known_args()

    # Encrypt params
    global keyvault_name, secret_tenant_sp, secret_client_sp, secret_sp, pk_secret, pk_pass_secret, pubk_secret
    keyvault_name = args.keyvault_name
    secret_tenant_sp = args.secret_tenant_sp
    secret_client_sp = args.secret_client_sp
    secret_sp = args.secret_sp
    pk_secret = args.pk_secret
    pk_pass_secret = args.pk_pass_secret
    pubk_secret = args.pubk_secret

    # Instantiate credential manager
    global cm
    cm = CredentialManager(keyvault_name, secret_tenant_sp, secret_client_sp, secret_sp, pubk_secret, pk_secret, pk_pass_secret)

    # Diarization parameters
    global msdd_model, msdd_cfg, input_audio_path, output_diar_path
    input_audio_path = args.input_audio_path
    output_diar_path = args.output_diar_path

    # Folder structure
    Path('./input').mkdir(parents=True, exist_ok=True)
    Path('./decrypted_files').mkdir(parents=True, exist_ok=True)
    Path('./nemo_diar_output').mkdir(parents=True, exist_ok=True)
    Path(output_diar_path).mkdir(parents=True, exist_ok=True)

    # Config files
    query_parameters = {"downloadformat": "yaml"}
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/stt_aml_deploy/src/components/diar/src/input/diar_infer_meeting.yaml', params=query_parameters)
    with open("./input/diar_infer_telephonic.yaml", mode="wb") as f:
        f.write(response.content)
    response = requests.get('https://raw.githubusercontent.com/hedrergudene/asr-sd-pipeline/main/stt_aml_deploy/src/components/diar/src/input/diar_infer_meeting.yaml', params=query_parameters)
    with open("./input/diar_infer_meeting.yaml", mode="wb") as f:
        f.write(response.content)

    # Read NeMo MSDD configuration file
    round_digits = lambda number, digits: int(number*10**digits)/10**digits
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{args.event_type}.yaml')
    msdd_cfg.diarizer.clustering.parameters.max_num_speakers = args.max_num_speakers
    msdd_cfg.diarizer.vad.external_vad_manifest='./input/asr_vad_manifest.json'
    msdd_cfg.diarizer.asr.parameters.asr_based_vad = True
    msdd_cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [
        round_digits(8*args.min_window_length,2),
        round_digits(5*args.min_window_length,2),
        round_digits(3*args.min_window_length,2),
        round_digits(2*args.min_window_length,2),
        round_digits(args.min_window_length,2)
    ]
    msdd_cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [
        round_digits(8*args.min_window_length/2,3),
        round_digits(5*args.min_window_length/2,3),
        round_digits(3*args.min_window_length/2,3),
        round_digits(2*args.min_window_length/2,3),
        round_digits(args.min_window_length/2,3)
    ]
    msdd_cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [args.overlap_threshold]
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    # Initialize NeMo MSDD diarization model
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)


def run(mini_batch):

    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        if not re.search(r'(.*?).*_nfa\.json\.pgp$', str(pathdir)):
            log.info(f"File {str(pathdir)} does not contain metadata from NFA. Skipping...")
            continue
        input_path = '/'.join(str(pathdir).split('/')[:-1])
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        fn = re.findall('(.*?)_nfa', fn)[0] # remove '_prep' from filename to get unique_id
        log.info(f"Processing file {fn}:")
        # Read word-level transcription to fetch timestamps
        cm.decrypt(input_path, './decrypted_files', f"{fn}_nfa.json.pgp")
        with open(f"./decrypted_files/{fn}_nfa.json", 'r', encoding='utf-8') as f:
            x = json.load(f)['segments']
        # Ensure audio contains activity
        if len(x)==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"./decrypted_files/{fn}_diar.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': [], # List of dictionaries with keys 'start', 'end'
                        'segments': []
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            cm.encrypt('./decrypted_files', output_diar_path, f"{fn}_diar.json", True)
            continue
        word_ts = [[w['start'], w['end']] for segment in x for w in segment['words']]

        #
        # Decrypt (if needed)
        #
        if os.path.isfile(f"{input_audio_path}/{fn}.wav.pgp"):
            log.info(f"Decrypt:")
            cm.decrypt(input_path, './decrypted_files', f"{fn}.wav.pgp")
            filepath = f"./decrypted_files/{fn}.wav"
        elif os.path.isfile(f"{input_audio_path}/{fn}.wav"):
            filepath = f"{input_audio_path}/{fn}.wav" 
        
        # Create ./input/asr_vad_manifest.json
        create_asr_vad_config(x, filepath)

        #
        # Speaker diarization
        #
        log.info(f"Run diarization")
        diar_time = time.time()
        create_msdd_config([filepath]) # initialise msdd cfg
        msdd_model.audio_file_list = [filepath] # update audios list
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, {fn:word_ts})
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")
        # Process diarization output
        log.info(f"Save outputs")
        segments = process_diar_output(diar_hyp)[fn]
        with open(os.path.join(f"./decrypted_files/{fn}_diar.json"), 'w', encoding='utf8') as f:
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
        cm.encrypt('./decrypted_files', output_diar_path, f"{fn}_diar.json", True)

        log.info(f"Cleanup resources")
        delete_files_in_directory_and_subdirectories('./decrypted_files')
        delete_files_in_directory_and_subdirectories('./nemo_diar_output')

    return mini_batch


def shutdown():
    cm.enable_secret(cm.sp_login(), pk_secret, False)
    cm.enable_secret(cm.sp_login(), pk_pass_secret, False)
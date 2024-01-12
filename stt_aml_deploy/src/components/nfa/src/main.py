# Libraries
import sys
import subprocess as sp
import logging as log
from pathlib import Path
import os
import re
import json
import time
from typing import List, Dict, Tuple, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import pgpy
import json
import argparse


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
            secret_sp:str=None
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


    def encrypt(
            self,
            input_path:str,
            output_path:str,
            filenames:List[str],
            puk_secret_name:str,
            remove_input:bool=False,
            secret_client:SecretClient=None
    ) -> None:
        # Check input is a list
        if isinstance(filenames, str):
            filenames = [filenames]
        # Default login
        if ((self.login!='default') | (secret_client is None)):
            secret_client = self.default_login()
        # Fetch secret
        secret_value = secret_client.get_secret(puk_secret_name).value
        # Import public key from PGP
        public_key, _ = pgpy.PGPKey.from_blob(secret_value)
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
            encrypted_message = public_key.encrypt(message)
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
            prk_secret_name:str,
            prk_password_secret_name:str,
            remove_input:bool=False,
            secret_client:SecretClient=None
    ) -> None:
        # Check input is a list
        if isinstance(filenames, str):
            filenames = [filenames]
        # Service principal login
        if ((self.login!='sp') | (secret_client is None)):
            secret_client = self.sp_login()
        # Retrieve pk secrets
        pk_secret_value = self.fetch_disabled_secret(secret_client, prk_secret_name)
        pk_pass_secret_value = self.fetch_disabled_secret(secret_client, prk_password_secret_name)
        # Fetch pk key
        private_key, _ = pgpy.PGPKey.from_blob(pk_secret_value)
        # Loop
        for filename in filenames:
            input_filepath = os.path.join(input_path, filename)
            folder_path, fn, ext = self.get_file_attr(input_filepath)
            if ext!='.pgp':
                log.warning(f"File {fn} is already decrypted. Skipping...")
                continue
            with private_key.unlock(pk_pass_secret_value) as ukey:
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
        if self.login!='sp':
            secret_client = self.sp_login()
        secret_client.update_secret_properties(secret_name, enabled=enable)
    

    def fetch_disabled_secret(
            self,
            secret_client:SecretClient,
            secret_name:str
    ) -> str:
        if self.login!='sp':
            secret_client = self.sp_login()
        self.enable_secret(secret_client, secret_name, True)
        secret_value = secret_client.get_secret(secret_name).value
        self.enable_secret(secret_client, secret_name, False)
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
    parser.add_argument("--input_asr_path", type=str)
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--secret_tenant_sp", type=str)
    parser.add_argument("--secret_client_sp", type=str)
    parser.add_argument("--secret_sp", type=str)
    parser.add_argument("--pk_secret", type=str)
    parser.add_argument("--pk_pass_secret", type=str)
    parser.add_argument("--pubk_secret", type=str)
    parser.add_argument("--nfa_model_name", type=str, default='stt_es_fastconformer_hybrid_large_pc')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_fa_path", type=str)
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

    # Params
    global input_audio_path, nfa_model_name, batch_size, output_fa_path
    input_audio_path = args.input_audio_path
    nfa_model_name = args.nfa_model_name
    batch_size = args.batch_size
    output_fa_path = args.output_fa_path

    # Folder structure
    Path('./decrypted_files').mkdir(parents=True, exist_ok=True)
    Path('./NeMo').mkdir(parents=True, exist_ok=True)
    Path('./input').mkdir(parents=True, exist_ok=True)
    Path('./nemo_nfa_output').mkdir(parents=True, exist_ok=True)
    Path(output_fa_path).mkdir(parents=True, exist_ok=True)

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


def run(mini_batch):

    # Instantiate credential manager
    cm = CredentialManager(keyvault_name, secret_tenant_sp, secret_client_sp, secret_sp)

    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        if not re.search(r'(.*?).*_asr\.json\.pgp$', str(pathdir)):
            log.info(f"File {str(pathdir)} does not contain metadata from ASR. Skipping...")
            continue
        input_path = '/'.join(str(pathdir).split('/')[:-1])
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        fn = re.findall('(.*?)_asr', fn)[0] # remove '_prep' from filename to get unique_id
        log.info(f"Processing file {fn}:")

        # Read word-level transcription to fetch timestamps
        cm.decrypt(input_path, './decrypted_files', f"{fn}_asr.json.pgp", pk_secret, pk_pass_secret)
        with open(f"./decrypted_files/{fn}_asr.json", 'r', encoding='utf-8') as f:
            asr_dct = json.load(f)
        # Ensure audio contains activity
        if len(asr_dct['segments'])==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"./decrypted_files/{fn}_nfa.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': [], # List of dictionaries with keys 'start', 'end'
                        'segments': []
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            cm.encrypt('./decrypted_files', output_fa_path, f"{fn}_nfa.json", pubk_secret, True)
            continue

        #
        # Decrypt (if needed)
        #
        if os.path.isfile(f"{input_audio_path}/{fn}.wav.pgp"):
            log.info(f"Decrypt:")
            cm.decrypt(input_path, './decrypted_files', f"{fn}.wav.pgp", pk_secret, pk_pass_secret)
            filepath = f"./decrypted_files/{fn}.wav"
        elif os.path.isfile(f"{input_audio_path}/{fn}.wav"):
            filepath = f"{input_audio_path}/{fn}.wav"

        # Create config
        create_nfa_config(asr_dct['segments'], filepath)

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
                f'pretrained_name="{nfa_model_name}"',
                'manifest_filepath="./input/nfa_manifest.jsonl"',
                'output_dir="./nemo_nfa_output"',
                f'batch_size={batch_size}',
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
            with open(f"./decrypted_files/{fn}_nfa.json", 'w', encoding='utf8') as f:
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
            cm.encrypt('./decrypted_files', output_fa_path, f"{fn}_nfa.json", pubk_secret, True)
        elif ((result.returncode==0) & (asr_dct['segments'][0].get('words') is None)):
            log.info(f"Alignment run successfully. Including word-level timestamps.")
            # Update timestamps from both segment-level and word-level information
            segments = process_nfa_output(fn)
            # Process output
            with open(f"./decrypted_files/{fn}_nfa.json", 'w', encoding='utf8') as f:
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
            cm.encrypt('./decrypted_files', output_fa_path, f"{fn}_nfa.json", pubk_secret, True)
        else:
            # Update timestamps from both segment-level and word-level information
            log.info(f"Alignment run successfully. Updating word-level timestamps.")
            segments = process_nfa_output(fn)

            # Keep confidence results from ASR
            for asr_seg, nfa_seg in zip(asr_dct['segments'], segments):
                for asr_word, nfa_word in zip(asr_seg['words'], nfa_seg['words']):
                    nfa_word['confidence'] = asr_word['confidence']

            # Process output
            with open(f"./decrypted_files/{fn}_nfa.json", 'w', encoding='utf8') as f:
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
            cm.encrypt('./decrypted_files', output_fa_path, f"{fn}_nfa.json", pubk_secret, True)
        log.info(f"Cleanup resources")
        delete_files_in_directory_and_subdirectories('./decrypted_files')
        delete_files_in_directory_and_subdirectories('./nemo_nfa_output')

    return mini_batch
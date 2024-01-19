# Libraries
import argparse
import sys
import logging as log
from pathlib import Path
import os
import re
import time
import json
from typing import List, Dict, Tuple, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import pgpy
import torch
from faster_whisper import WhisperModel


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
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--secret_tenant_sp", type=str)
    parser.add_argument("--secret_client_sp", type=str)
    parser.add_argument("--secret_sp", type=str)
    parser.add_argument("--pk_secret", type=str)
    parser.add_argument("--pk_pass_secret", type=str)
    parser.add_argument("--pubk_secret", type=str)
    parser.add_argument("--whisper_model_name", type=str, default='large-v3')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--word_level_timestamps", type=bool, default=True)
    parser.add_argument("--condition_on_previous_text", type=bool, default=True)
    parser.add_argument("--compute_type", type=str, default='float16')
    parser.add_argument("--language_code", type=str, default='es')
    parser.add_argument("--output_asr_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Folder structure
    Path('./decrypted_files').mkdir(parents=True, exist_ok=True)

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

    # ASR params
    global beam_size, word_level_timestamps, condition_on_previous_text, language_code, output_asr_path
    beam_size = args.beam_size
    word_level_timestamps = args.word_level_timestamps
    condition_on_previous_text = args.condition_on_previous_text
    language_code = args.language_code
    output_asr_path = args.output_asr_path

    # Folder structure
    Path(output_asr_path).mkdir(parents=True, exist_ok=True)

    # ASR models
    global whisper_model
    whisper_model = WhisperModel(
        model_size_or_path=args.whisper_model_name,
        device=device,
        compute_type=args.compute_type,
        cpu_threads=os.cpu_count(),
        num_workers=args.num_workers
    )


def run(mini_batch):

    for elem in mini_batch:
        # Read file and filter if necessary (we are only looking for files with pattern '(.*?)_metadata.json')
        pathdir = Path(elem)
        if not re.search(r'(.*?).*_prep\.json\.pgp$', str(pathdir)):
            log.info(f"File {str(pathdir)} does not contain metadata from preprocessing. Skipping...")
            continue
        input_path = '/'.join(str(pathdir).split('/')[:-1])
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        fn = re.findall('(.*?)_prep', fn)[0] # remove '_prep' from filename to get unique_id

        # Fetch metadata
        log.info(f"Processing file {fn}:")
        cm.decrypt(input_path, './decrypted_files', f"{fn}_prep.json.pgp")
        with open(f'./decrypted_files/{fn}_prep.json', 'r') as f:
            metadata_dct = json.load(f)

        # Ensure audio contains activity
        if len(metadata_dct['vad_timestamps'])==0:
            log.info(f"Audio {fn} does not contain any activity. Generating dummy metadata:")
            with open(f"./decrypted_files/{fn}_asr.json", 'w') as f:
                json.dump(
                    {
                        'vad_timestamps': metadata_dct['vad_timestamps'], # List of dictionaries with keys 'start', 'end'
                        'segments': []
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            cm.encrypt('./decrypted_files', output_asr_path, f"{fn}_asr.json", True)
            continue

        #
        # Decrypt (if needed)
        #
        if os.path.isfile(f"{input_path}/{fn}.wav.pgp"):
            log.info(f"Decrypt:")
            cm.decrypt(input_path, './decrypted_files', f"{fn}.wav.pgp")
            filepath = f"./decrypted_files/{fn}.wav"
        elif os.path.isfile(f"{input_path}/{fn}.wav"):
            filepath = f"{input_path}/{fn}.wav"

        #
        # Transcription
        #
        log.info(f"\tASR:")
        transcription_time = time.time()
        segments, _ = whisper_model.transcribe(
            filepath,
            beam_size=beam_size,
            language=language_code,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=False,
            word_timestamps=word_level_timestamps
        )

        if word_level_timestamps:
            segs = []
            end_repl = lambda text: re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
            start_repl = lambda text: re.sub(r'([¿¡"])\s+', r'\1', text)
            for x in segments:
                words = []
                if len(x.words)==0: continue # So that global stats basen on word ts are not messed up
                for word in x.words:
                    words.append(
                       {
                          'start':word.start,
                          'end':word.end,
                          'text':end_repl(start_repl(word.word.strip())),
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
            segs = [{'start': x.start, 'end': x.end, 'text': end_repl(start_repl(x.text.strip()))} for x in segments]
        transcription_time = time.time()-transcription_time
        log.info(f"\t\tTranscription time: {transcription_time}")

        # Build metadata
        mtd = {
            "transcription_time": transcription_time
        }
        # Save output
        with open(f'./decrypted_files/{fn}_asr.json', 'w', encoding='utf8') as f:
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
        cm.encrypt('./decrypted_files', output_asr_path, f"{fn}_asr.json", True)

        # Cleanup resources
        delete_files_in_directory_and_subdirectories('./decrypted_files')

    return mini_batch


def shutdown():
    cm.enable_secret(cm.sp_login(), pk_secret, False)
    cm.enable_secret(cm.sp_login(), pk_pass_secret, False)
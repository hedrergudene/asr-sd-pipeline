# Libraries
import argparse
import sys
import logging as log
from transformers import pipeline
import torch
import re
import json
import os
import re
import time
import bisect
from typing import List, Dict, Tuple, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import pgpy
import pymongo
from datetime import datetime
from pathlib import Path

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Helper functions to align sentences with punctuation signs
def get_word_ts_anchor(s, e, option="mid"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s

def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="mid"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wc, wrd = (
            wrd_dict["start"],
            wrd_dict["end"],
            wrd_dict["confidence"],
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {
                "word": wrd,
                "start_time": ws,
                "end_time": we,
                "confidence": wc,
                "speaker": sp
            }
        )
    return wrd_spk_mapping

def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words, sentence_ending_punctuations):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

def get_last_word_idx_of_sentence(word_idx, word_list, max_words, sentence_ending_punctuations):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list)
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )

def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence, sentence_ending_punctuations
):
    is_word_sentence_end = (
        lambda x: ((x >= 0)
        & (word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations))
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence, sentence_ending_punctuations
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1, sentence_ending_punctuations
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list

def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    cf = []
    words = []

    snt = {
        "speaker": f"Speaker {spk}",
        "start_time": s,
        "end_time": e,
        "confidence": 0,
        "text": "",
        "words": []
    }

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": wrd_dict["start_time"],
                "end_time": wrd_dict["end_time"],
                "confidence": 0,
                "text": "",
                "words": [],
            }
            cf = []
            words = []
        else:
            snt["end_time"] = wrd_dict["end_time"]
            cf.append(wrd_dict['confidence'])
            snt['confidence'] = sum(cf)/len(cf)
            words.append(wrd_dict)
            snt['words'] = words
        snt["text"] += wrd + " "
        prev_spk = spk
    snt['text'] = snt['text'].strip()
    snts.append(snt)
    return snts


# Class to align VAD timestamps
class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: List[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float) -> int:
        sample = int(time * self.sampling_rate)
        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )


# Punctuation model
class PunctuationModel():
    def __init__(
            self,
            model:str = "oliverguhr/fullstop-punctuation-multilang-large"
    ) -> None:        
        if torch.cuda.is_available():
            self.pipe = pipeline("ner",model, aggregation_strategy="none", device=0)
        else:
            self.pipe = pipeline("ner",model, aggregation_strategy="none")

    def preprocess(self,text):
        #remove markers except for markers in numbers 
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)","",text) 
        #todo: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self,text):        
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)
        
    def overlap_chunks(self,lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n-stride):
                yield lst[i:i + n]

    def predict(self,words, chunk_size=80, overlap=5):
        if len(words) <= chunk_size:
            overlap = 0

        batches = list(self.overlap_chunks(words, chunk_size, overlap))

        # if the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= overlap:
            batches.pop()

        tagged_words = []     
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]: 
                overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)
            assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"
                
            char_index = 0
            result_index = 0
            for word in batch[:len(batch)-overlap]:
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = "0"
                while result_index < len(result) and char_index > result[result_index]["end"] :
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1                        
                tagged_words.append([word,label, score])
        
        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self,prediction):
        result = ""
        for word, label, _ in prediction:
            result += word
            if label == "0":
                result += " "
            if label in ".,?-:":
                result += label+" "
        return result.strip()


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
    parser.add_argument("--input_diar_path", type=str)
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
    parser.add_argument("--ner_chunk_size", type=int, default=80)
    parser.add_argument("--ner_stride", type=int, default=5)
    parser.add_argument("--max_words_in_sentence", type=int, default=40)
    parser.add_argument("--output_sm_path", type=str)
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

    # Params
    global input_diar_path, max_words_in_sentence, output_sm_path
    input_diar_path = args.input_diar_path
    max_words_in_sentence = args.max_words_in_sentence
    output_sm_path = args.output_sm_path

    # Folder structure
    Path('./decrypted_files').mkdir(parents=True, exist_ok=True)
    Path(output_sm_path).mkdir(parents=True, exist_ok=True)

    # Punctuation model
    global punct_model, ending_puncts, model_puncts, ner_chunk_size, ner_stride
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    ending_puncts = '.?!'
    model_puncts = ".,;:!?"
    ner_chunk_size = args.ner_chunk_size
    ner_stride = args.ner_stride

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

    # Timestamp
    global ts
    ts = str(datetime.now())


def run(mini_batch):

    for elem in mini_batch: # mini_batch on ASR files
        # Read file
        pathdir = Path(elem)
        if not re.search(r'(.*?).*_nfa\.json\.pgp$', str(pathdir)):
            log.info(f"File {str(pathdir)} does not contain metadata from diarization. Skipping...")
            continue
        input_path = '/'.join(str(pathdir).split('/')[:-1])
        fn, _ = os.path.splitext(str(pathdir).split('/')[-1])
        fn = re.findall('(.*?)_nfa', fn)[0] # remove '_nfa' from filename to get unique_id

        # Process
        log.debug(f"Processing file {fn}:")
        cm.decrypt(input_path, './decrypted_files', f"{fn}_nfa.json.pgp")
        with open(f"./decrypted_files/{fn}_nfa.json", 'r', encoding='utf8') as f:
            asr_dct = json.load(f)
        
        # If file contains no segments, jump to the next one generating dummy metadata
        if len(asr_dct['segments'])==0:
            log.debug(f"Audio {fn} does not contain segments. Dumping dummy file and skipping:")
            # Save output
            with open(
                f'./decrypted_files/{fn}.json',
                'w',
                encoding='utf8'
            ) as f:
                json.dump(
                    {
                    'unique_id': fn,
                    'segments': []
                },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            cm.encrypt('./decrypted_files', output_sm_path, f"{fn}.json", True)
            # Generate record in cosmosDB
            cosmosdb_client.update_one(
                {"_id": fn}, {"$set": {"timestamp": ts}}, upsert=True
            )
            continue
        asr_input = [w for s in asr_dct['segments'] for w in s['words']]

        # Diarization metadata
        cm.decrypt(input_path, './decrypted_files', f"{fn}_diar.json.pgp")
        with open(f"./decrypted_files/{fn}_diar.json", 'r', encoding='utf8') as f:
            diar_dct = json.load(f)
        diar_input = [[s['start'], s['end'], s['speaker']] for s in diar_dct['segments']]

        # Get labels for each piece of text from ASR
        sm_time = time.time()
        wsm = get_words_speaker_mapping(asr_input, diar_input)
        words_list = list(map(lambda x: x["word"], wsm))
        try:
            labled_words = punct_model.predict(words_list, ner_chunk_size, ner_stride)
        except:
            log.warning(f"Raised error using chunk_size={ner_chunk_size}. Retrying with half chunk size.")
            try:
                labled_words = punct_model.predict(words_list, ner_chunk_size//2, ner_stride)
            except:
                log.warning(f"Raised error using chunk_size={ner_chunk_size//2}. Retrying with half chunk size.")
                try:
                    labled_words = punct_model.predict(words_list, ner_chunk_size//4, ner_stride)
                except:
                    log.warning(f"Raised error using chunk_size={ner_chunk_size//4}. Retrying with half chunk size.")
                    labled_words = punct_model.predict(words_list, ner_chunk_size//8, ner_stride)
        # Acronyms handling
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
        wsm_final = get_realigned_ws_mapping_with_punctuation(wsm, max_words_in_sentence, ending_puncts)
        ssm = get_sentences_speaker_mapping(wsm_final, diar_input)
        sm_time = time.time() - sm_time
        log.info(f"\tSentence-mapping time: {sm_time}")

        #
        # Adjust timestamps with VAD chunks
        #
        log.info('\tMapping VAD timestamps with transcription')
        ts_map = SpeechTimestampsMap(asr_dct['vad_timestamps'], 16000)
        for segment in ssm:
                words = []
                for word in segment['words']:
                    # Ensure the word start and end times are resolved to the same chunk.
                    middle = (word['start_time'] + word['end_time']) / 2
                    chunk_index = ts_map.get_chunk_index(middle)
                    word['start_time'] = ts_map.get_original_time(word['start_time'], chunk_index)
                    word['end_time'] = ts_map.get_original_time(word['end_time'], chunk_index)
                    words.append(word)

                    segment['start_time'] = words[0]['start_time']
                    segment['end_time'] = words[-1]['end_time']
                    segment['words'] = words


        # Save output
        with open(
            os.path.join(
                './decrypted_files',
                f"{fn}.json"
            ),
            'w',
            encoding='utf8'
        ) as f:
            json.dump(
                {
                'unique_id': fn,
                'duration': asr_dct['duration'],
                'processing_time': {
                    **asr_dct['metadata'],
                    **diar_dct['metadata'],
                    **{
                        'sentence_mapping_time': sm_time
                    }
                },
                'segments': ssm
            },
                f,
                indent=4,
                ensure_ascii=False
            )

        # Generate record in cosmosDB
        cosmosdb_client.update_one(
            {"_id": fn},
            {
                "$set": {
                    "timestamp": ts,
                    'duration': asr_dct['duration'],
                    'processing_time': {
                        **asr_dct['metadata'],
                        **diar_dct['metadata'],
                        **{
                            'sentence_mapping_time': sm_time
                        }
                    }
                }
            }, upsert=True
        )
        
        # Decrypt output
        cm.encrypt('./decrypted_files', output_sm_path, f"{fn}.json", True)

        # Cleanup resources
        log.info(f"Cleanup resources")
        delete_files_in_directory_and_subdirectories('./decrypted_files')

    return mini_batch


def shutdown():
    cm.enable_secret(cm.sp_login(), pk_secret, False)
    cm.enable_secret(cm.sp_login(), pk_pass_secret, False)
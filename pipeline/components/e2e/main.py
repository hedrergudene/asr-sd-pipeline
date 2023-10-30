# Requierments
## Essentials
import logging as log
import os
import sys
import subprocess as sp
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
from transformers import pipeline
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from omegaconf import OmegaConf
## Azure
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
## Custom utilities
from src.sentence_mapping_model import PunctuationModel
from src.utils import *
from src.vad import SpeechTimestampsMap

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
    

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    keyvault_name,
    secret_name,
    vad_threshold,
    min_speech_duration_ms,
    min_silence_duration_ms,
    speech_pad_ms,
    use_onnx_vad,
    demucs_model,
    asr_model_name,
    asr_compute_type,
    asr_chunk_length_s,
    asr_batch_size,
    fa_model_name,
    fa_batch_size,
    event_type,
    max_num_speakers,
    max_words_in_sentence,
    output_path
):
    # Create output paths
    Path('./decrypted_audios').mkdir(parents=True, exist_ok=True)
    Path('./prep_audios').mkdir(parents=True, exist_ok=True)
    Path('./nemo_nfa_output').mkdir(parents=True, exist_ok=True)
    Path('./nemo_diar_output').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

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

    #ASR model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipeline("automatic-speech-recognition",
                    asr_model_name,
                    torch_dtype=torch.float16 if asr_compute_type=='float16' else torch.int8,
                    device=device)
    pipe.model = pipe.model.to_bettertransformer()

    # Diarization model
    ## Read NeMo MSDD configuration file
    msdd_cfg = OmegaConf.load(f'./input/diar_infer_{event_type}.yaml')
    msdd_cfg.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
    msdd_cfg.diarizer.asr.parameters.asr_based_vad = True
    create_msdd_config(['sample_audio.wav']) # initialise msdd cfg
    ## Initialize NeMo MSDD diarization model
    msdd_model = OfflineDiarWithASR(msdd_cfg.diarizer)

    # Sentence mapping model
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    sentence_ending_punctuations = '.?!'
    model_puncts = ".,;:!?"


    for filename in os.listdir(input_path):

        #
        # Audio preprocessing
        #
        fn, ext = os.path.splitext(filename)
        log.info(f"File {fn}:")
        # Decrypt file
        with open(f"{input_path}/{filename}", 'rb') as f:
            encrypted_bytes = f.read()
        decrypted_bytes = my_fernet.decrypt(encrypted_bytes)
        # Save decrypted data
        with open(f'./decrypted_audios/{filename}', 'wb') as fd:
            fd.write(decrypted_bytes)
        # Standarise format
        log.info(f"\tRun preprocessing")
        prep_time = time.time()
        preprocess_audio('./decrypted_audios', './prep_audios', filename)
        # VAD
        wav = read_audio(f"./prep_audios/{filename}", sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            threshold=vad_threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
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
            f"./prep_audios/{fn}_vad_demucs_mono.wav"
        ]
        out = subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        if out.returncode!=0:
            raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")
        prep_time = time.time() - prep_time
        log.info(f"\tPrepocessing time: {prep_time}")
        # Save audio filepath
        clean_audio_filepath = f"./prep_audios/{fn}_vad_demucs_mono.wav"

        #
        # ASR
        #
        log.info(f"\tRun ASR")
        asr_time = time.time()
        segments = pipe(
            clean_audio_filepath,
            chunk_length_s=asr_chunk_length_s,
            batch_size=asr_batch_size,
            return_timestamps=True
        )['chunks'] # dictionary with keys 'text' and 'timestamps', being the latter a tuple with format (start_ts, end_ts)
        segments = [{'start': sg['timestamp'][0], 'end': sg['timestamp'][1], 'text': sg['text']} for sg in segments]
        asr_time = time.time() - asr_time
        log.info(f"\tASR time: {asr_time}")

        #
        # Forced alignment (Viterbi algorithm)
        #

        # Create ./input/nfa_manifest.json
        create_nfa_config(segments, clean_audio_filepath)
        # Run script
        log.info(f"\tRun alignment")
        fa_time = time.time()
        result = sp.run(
            [
                sys.executable,
                'NeMo/tools/nemo_forced_aligner/align.py',
                f'pretrained_name={fa_model_name}',
                'manifest_filepath="./input/nfa_manifest.jsonl"',
                'output_dir="./nemo_nfa_output/"',
                f'batch_size={fa_batch_size}',
                'additional_segment_grouping_separator="|"'
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='xmlcharrefreplace'
        )
        # Check return code
        if result.returncode!=0:
            log.error(f"Alignment raised an exception: {result.stderr}")
            raise RuntimeError(f"Alignment raised an exception: {result.stderr}")
        # Get runtime
        fa_time = time.time() - fa_time
        log.info(f"\tForced alignment time: {fa_time}")
        # Get word-level timestamps
        word_level_ts, word_ts = process_nfa_output(f"{fn}_vad_demucs_mono")

        #
        # Speaker diarization
        #
        log.info(f"\tRun diarization")
        diar_time = time.time()
        # Create ./input/asr_vad_manifest.json
        create_asr_vad_config(word_level_ts, clean_audio_filepath)
        # Create diarization config
        create_msdd_config([clean_audio_filepath]) # initialise msdd cfg
        msdd_model.audio_file_list = [clean_audio_filepath] # update audios list
        # Run diarization
        diar_hyp, _ = msdd_model.run_diarization(msdd_cfg, {f"{fn}_vad_demucs_mono":word_ts})
        diar_time = time.time() - diar_time
        log.info(f"\tDiarization time: {diar_time}")

        #
        # Sentence alignment
        #
        log.info(f"\tRun Sentence mapping")
        sm_time = time.time()
        asr_input = [w for s in word_level_ts for w in s['words']]
        diar_input = [[s['start'], s['end'], s['speaker']] for s in process_diar_output(diar_hyp)[f"{fn}_vad_demucs_mono"]]
        wsm = get_words_speaker_mapping(asr_input, diar_input)
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list)
        # Acronyms handling
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in sentence_ending_punctuations
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
        wsm_final = get_realigned_ws_mapping_with_punctuation(wsm, max_words_in_sentence, sentence_ending_punctuations)
        ssm = get_sentences_speaker_mapping(wsm_final, diar_input)
        sm_time = time.time() - sm_time
        log.info(f"\tSentence-mapping time: {sm_time}")

        #
        # Adjust timestamps with VAD chunks
        #
        log.info('\tMapping VAD timestamps with transcription')
        ts_map = SpeechTimestampsMap(speech_timestamps, 16000)
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

        #
        # Save output
        #
        ## Define dictionary
        output = {
                'unique_id': fn,
                'duration': audio_length_s,
                'processing_time': {
                    'preprocessing_time': prep_time,
                    'asr_time': asr_time,
                    'forced_alignment_time': fa_time,
                    'diarization_time': diar_time,
                    'sentence_mapping_time': sm_time
                },
                'segments': ssm
            }
        ## Save the file decrypted (to be read as bytes)
        with open(os.path.join(output_path,f"{fn}.json"), 'w', encoding='utf8') as f:
            json.dump(
                output,
                f,
                indent=4,
                ensure_ascii=False
            )
        ## Read it as bytes sequence
        with open(os.path.join(output_path,f"{fn}.json"), 'rb') as f:
            decrypted_bytes = f.read()
        ## Encript data
        encrypted_bytes = my_fernet.encrypt(decrypted_bytes)
        # Save encrypted data
        with open(os.path.join(output_path,f"{fn}.json"), 'wb') as fd:
            fd.write(encrypted_bytes)

        # Cleanup resources
        delete_files_in_directory_and_subdirectories('./decrypted_audios')
        delete_files_in_directory_and_subdirectories('./prep_audios')
        delete_files_in_directory_and_subdirectories('./nemo_nfa_output')
        delete_files_in_directory_and_subdirectories('./nemo_diar_output')

if __name__=="__main__":
    fire.Fire(main)
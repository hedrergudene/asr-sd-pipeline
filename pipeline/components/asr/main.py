# Requierments
import logging as log
import time
import subprocess
import os
import sys
import json
from pathlib import Path
import torch
import librosa
import soundfile as sf
from faster_whisper import WhisperModel
import fire


# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Helper function to fetch files
def get_file(f):
    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            log.error(f"More than one file was found in directory: {','.join(files)}.")
            return (f"More than one file was found in directory: {','.join(files)}.", 500)

# Helper function to get CUDA compute capability
def get_cuda_compute():
    output = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"], 
                            stdout=subprocess.PIPE, 
                            text=True
    )
    return float(output.stdout.split('\n')[1])

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
    whisper_model_name,
    num_workers,
    beam_size,
    vad_threshold,
    min_speech_duration_ms,
    min_silence_duration_ms,
    word_level_timestamps,
    condition_on_previous_text,
    compute_type,
    language_code,
    output_path
):
    # Folder structure
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ASR model
    whisper_model = WhisperModel(
        model_size_or_path = whisper_model_name,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers
    )

    # Set up input
    f = Path(input_path)
    files = list(f.iterdir())

    for pathdir in files:
        # Read file
        filename, extension = os.path.splitext(str(pathdir).split('/')[-1])
        if extension not in ['.wav', '.mp3']:
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
        #sf.write(f'./input_audios/{filename}.wav', signal, 16000, 'PCM_24') # save in tmp path as 16kHz, mono
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
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=True,
            word_timestamps=word_level_timestamps,
            vad_parameters=dict(
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        )

        if word_level_timestamps:
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
        else:
            segs = [{'start': x.start, 'end': x.end, 'text': x.text.strip()} for x in segments]
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

if __name__=="__main__":
    fire.Fire(main)
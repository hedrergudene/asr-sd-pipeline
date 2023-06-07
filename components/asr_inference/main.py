# Requierments
import logging as log
import time
import subprocess
import os
import pandas as pd
import sys
from pathlib import Path
import torch
from faster_whisper import WhisperModel
import whisperx
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


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    whisper_model_name,
    num_workers,
    beam_size,
    vad_threshold,
    min_speech_duration_ms,
    min_silence_duration_ms,
    language_code,
    fp16,
    output_path
):
    # Create output path and auxiliary folders
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Check cuda capabilities
    if ((torch.cuda.is_available()) & (get_cuda_compute()<7.5)):
        raise Exception(f"Nvidia CUDA compute capabilities are below 7.5 ({get_cuda_compute()}), minimum threshold for Turing tensor cores.")
    # Fetch input paths from previous component
    input_filepath = get_file(input_path)
    df = pd.read_csv(input_filepath, index_col=[0])
    log.info(f"Dataframe:\n{df.head()}")
    pathdirs = df['paths'].values
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info(f"Device: {device}")
    whisper_model = WhisperModel(
        whisper_model_name,
        num_workers=num_workers,
        device=device,
        compute_type="float16" if fp16 else "int8_float16"
    )
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    
    # Loop
    for pathdir in pathdirs:
        filename = pathdir.split('/')[-1].split('.')[0]
        log.info(f"Processing file: {pathdir}")
        # Transcription
        transcription_time = time.time()
        segments, _ = whisper_model.transcribe(
            input_filepath,
            beam_size=beam_size,
            language=language_code,
            vad_filter=True,
            vad_parameters=dict(
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        )
        transcription_time = time.time()-transcription_time
        log.info(f"\tTranscription time: {transcription_time}")
        # Alignment
        alignment_time = time.time()
        result_aligned = whisperx.align(
            [{'start':x.start, 'end':x.end, 'text':x.text} for x in segments],
            alignment_model,
            metadata,
            pathdir,
            device
        )
        alignment_time = time.time() - alignment_time
        log.info(f"\tAlignment time: {alignment_time}")
        # Save file
        pd.DataFrame([{'start':x['start'], 'end':x['end'], 'text':x['text']} for x in result_aligned['word_segments']]).to_csv(os.path.join('./outputs', filename+'.csv'))
        log.info(f"\tTotal processing time: {transcription_time+alignment_time}")
    # Generate output
    os.system(f"rar a {os.path.join(output_path,'asr.rar')} ./outputs")

if __name__=="__main__":
    fire.Fire(main)
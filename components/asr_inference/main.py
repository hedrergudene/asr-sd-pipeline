# Requierments
import logging as log
import os
import pandas as pd
import sys
from pathlib import Path
import torch
import stable_whisper
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


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    whisper_model_name,
    vad_threshold,
    no_speech_threshold,
    language_code,
    output_path
):
    # Create output path and auxiliary folders
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Fetch input paths from previous component
    input_filepath = get_file(input_path)
    df = pd.read_csv(input_filepath, index_col=[0])
    log.info(f"Dataframe:\n{df.head()}")
    pathdirs = df['paths'].values
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info(f"Device: {device}")
    whisper_model = stable_whisper.load_model(whisper_model_name)
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    
    # Loop
    for pathdir in pathdirs:
        filename = pathdir.split('/')[-1].split('.')[0]
        log.info(f"Processing file: {pathdir}")
        # Transcription
        result = whisper_model.transcribe(
            pathdir,
            language=language_code,
            vad=True,
            vad_threshold=vad_threshold,
            #demucs=True,
            #demucs_output='sample.wav',
            no_speech_threshold=no_speech_threshold,
            regroup=True
        )
        # Alignment
        result_aligned = whisperx.align(
            [{'start':x.start, 'end':x.end, 'text':x.text} for x in result.segments],
            alignment_model,
            metadata,
            pathdir,
            device
        )
        # Save file
        pd.DataFrame([{'start':x['start'], 'end':x['end'], 'text':x['text']} for x in result_aligned['word_segments']]).to_csv(os.path.join('./outputs', filename+'.csv'))
    # Generate output
    os.system(f"rar a {os.path.join(output_path,'asr.rar')} ./outputs")

if __name__=="__main__":
    fire.Fire(main)
# Requierments
import logging as log
import time
from copy import deepcopy
import re
import subprocess
import os
import sys
import json
from pathlib import Path
import pandas as pd
import torch
import librosa
from num2words import num2words
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
    align_model_name,
    num_workers,
    beam_size,
    vad_threshold,
    min_speech_duration_ms,
    min_silence_duration_ms,
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
    alignment_model, metadata = whisperx.load_align_model(
            language_code = language_code,
            model_name = align_model_name,
            device=device
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
            vad_filter=True,
            word_timestamps=False,
            vad_parameters=dict(
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        )
        segs = [{'start': x.start, 'end': x.end, 'text': x.text} for x in segments]

        transcription_time = time.time()-transcription_time
        log.info(f"\t\tTranscription time: {transcription_time}")

        #
        # Forced alignment
        #
        log.info('Word-level forced alignment:')
        # Number decoding
        log.info(f"Digits decoding:")
        decoding_time = time.time()
        length, ents, seg_dct = 0, [], []
        for x in segs:
            text = x.get('text')
            # Matches that are related to numeric expressions in which numbers are divided by "." or ","
            text = re.sub(r'(\d) (?=[^\d\s])', r'\1', text)
            # Matches that are related to numeric expressions and special symbols
            regex = r'(?:[0-9]+(?:[-.,][0-9]+)*)'
            next_match = re.search(regex, text)
            while next_match is not None:
                ts, te, match = next_match.start(), next_match.end(), next_match.group()
                new_text = match.replace('-', ' ')
                new_text= ' '.join([num2words(z.replace(',',''), lang=language_code) for z in new_text.split(' ')])
                if text[next_match.start()-1:next_match.start()]=='$': # Dollars
                    new_text = new_text+' dólares' if language_code=='es' else new_text+' dollars'
                    ents.append({'text': new_text, 'pattern':'$'+match, 'start_idx': ts+length-1, 'end_idx': ts+(len(new_text)-1)+length})
                    text = text[:ts-1] + new_text + text[te:] # As percentage sign goes to the left
                elif text[next_match.start():next_match.start()+1]=='%': # Percentages
                    new_text = new_text+' por ciento' if language_code=='es' else new_text+' percent'
                    ents.append({'text': new_text, 'pattern':match+'%', 'start_idx': ts+length, 'end_idx': ts+(len(new_text)-1)+length+1})
                    text = text[:ts] + new_text + text[te+1:] # As percentage sign goes to the right
                else:
                    ents.append({'text': new_text, 'pattern':match, 'start_idx': ts+length, 'end_idx': ts+len(new_text)+length})
                    text = text[:ts] + new_text + text[te:]
                # Modify via shift entities that are after the chosen one
                shift = len(new_text) -len(match)
                for ent in [x for x in ents if (te+length)<=x['start_idx']]:
                    ent['start_idx'] += shift
                    ent['end_idx'] += shift
                # Restart loop
                next_match = re.search(regex, text)
            # Matches that are related to dots followed and preceded by characters
            regex=r'(?<=[A-Za-z])\.(?=[A-Za-z])'
            next_match = re.search(regex, text)
            new_text = ' punto ' if language_code=='es' else ' dot '
            while next_match is not None:
                ts, te, match = next_match.start(), next_match.end(), next_match.group()
                ents.append({'text': new_text, 'pattern':match, 'start_idx': ts+length, 'end_idx': ts+len(new_text)+length})
                text = text[:ts] + new_text + text[te:]
                # Modify via shift entities that are after the chosen one
                shift = len(new_text) - 1
                for ent in [x for x in ents if (te+length)<=x['start_idx']]:
                    ent['start_idx'] += shift
                    ent['end_idx'] += shift
                # Restart loop
                next_match = re.search(regex, text)
            # Append modified text
            seg_dct.append({'start': x['start'], 'end': x['end'], 'text': text})
            #! BIG UPDATE: ADD +1 TO ALIGN WITH JOINT TEXT
            length += len(text)+1
        decoding_time = time.time()-decoding_time
        log.info(f"\t\tDecoding time: {decoding_time}")

        #
        # Alignment
        #
        log.debug(f"\tAlignment:")
        alignment_time = time.time()
        result_aligned = whisperx.align(
            seg_dct,
            alignment_model,
            metadata,
            signal,
            device
        )
        alignment_time = time.time() - alignment_time
        log.debug(f"\t\tAlignment time: {alignment_time}")

        #
        # Digit (re)encoding
        #
        # Add ABSOLUTE start-end indices tokens to aligned output
        log.debug(f"\tDigits (re)encoding:")
        de_time = time.time()
        #! BIG UPDATE: Re-encoding based on word-level concatenation
        aligned_output, length = [], 0
        for blk in result_aligned['segments']:
            out = {
                'start': blk.get('start'),
                'end': blk.get('end'),
                'start_idx': length,
                'end_idx': len(blk['text'])+length,
                'text': blk.get('text'),
                'words':[]
            }
            shift = 0
            for word_dct in blk.get('words'):
                out['words'].append(
                    {
                        **word_dct,
                        **{
                            'start_idx': shift+length,
                            'end_idx': len(word_dct['word'])+shift+length
                        }
                    }
                )
                shift += len(word_dct['word'])+1
            length += shift
            aligned_output.append(out)
        # Merge information
        shift_global = 0
        ao = deepcopy(aligned_output)
        entities = deepcopy(ents)
        for seg in ao:
            shift_seg = 0
            seg['start_idx'], seg['end_idx'] = seg['start_idx']+shift_global, seg['end_idx']+shift_global
            seg['words'] = [{
                'word': x['word'],
                'start': x['start'],
                'end': x['end'],
                'score': x['score'],
                'start_idx': x['start_idx']+shift_global,
                'end_idx': x['end_idx']+shift_global
            } for x in seg['words' ]]
            ents_seg = [x for x in entities if ((seg['start_idx']<=x['start_idx']) & (seg['end_idx']>=x['end_idx']))]
            if len(ents_seg)>0:
                log.debug(f"Segment {seg['text']} with sidx {seg['start_idx']} and eidx {seg['end_idx']}. Entities {ents_seg}")
                ents_seg = [x.to_dict() for _, x in pd.DataFrame(ents_seg).sort_values(by='start_idx').iterrows()]
                for ent in ents_seg:
                    log.debug(f"\tAnalysing entity {ent['text']} matched with pattern {ent['pattern']} with sidx {ent['start_idx']} and eidx {ent['end_idx']}")
                    # Update entity positions...
                    ent = {'text': ent['text'], 'pattern': ent['pattern'], 'start_idx': ent['start_idx']+shift_seg, 'end_idx': ent['end_idx']+shift_seg}
                    # ...find all words between those positions and compute stats...
                    target_words_begin = [x for x in seg['words'] if ((x['end_idx']>=ent['start_idx']) & (x['start_idx']<=ent['end_idx']))]
                    agg_dct = pd.DataFrame(target_words_begin).agg({'start': 'min', 'end': 'max', 'score': 'mean', 'start_idx': 'min', 'end_idx': 'max'}).to_dict()
                    agg_dct['start_idx'] = int(ent['start_idx'])
                    agg_dct['word'] = re.sub(ent['text'], ent['pattern'],' '.join([x['word'] for x in target_words_begin]))
                    agg_dct['end_idx'] = ent['start_idx']+len(agg_dct['word'])
                    shift_ent = len(agg_dct['word'])-int(len(' '.join([x['word'] for x in target_words_begin])))
                    shift_seg += shift_ent
                    # ...remove decoded words...
                    for x in target_words_begin: seg['words'].remove(x)
                    # ...include new word...
                    seg['words'].append(agg_dct)
                    # ...update next values...
                    target_words_end = [x for x in seg['words'] if x['start_idx']>=ent['end_idx']]
                    for x in target_words_end:
                        seg['words'].remove(x)
                        seg['words'].append(
                            {
                                'word': x['word'],
                                'start': x['start'],
                                'end': x['end'],
                                'score': x['score'],
                                'start_idx': x['start_idx']+shift_ent,
                                'end_idx': x['end_idx']+shift_ent
                            }
                        )
                # ...and update global parameters
                seg['end_idx'] += len(' '.join([x['word'] for x in seg['words']])) - len(seg['text'])
                seg['text'] = ' '.join([x['word'] for x in seg['words']])
                entities = [
                    {
                        'text': x['text'],
                        'pattern': x['pattern'],
                        'start_idx': x['start_idx']+shift_seg,
                        'end_idx': x['end_idx']+shift_seg
                    } for x in entities
                ]
                shift_global += shift_seg
        # Cleanup indices
        for x in ao:
            x.pop('start_idx')
            x.pop('end_idx')
            for w in x['words']:
                w.pop('start_idx')
                w.pop('end_idx')
        de_time = time.time() - de_time
        log.debug(f"\t\tDigit (re)encoding time: {de_time}")
        # Build metadata
        mtd = {
            "preprocessing_time": prep_time,
            "transcription_time": transcription_time,
            "decoding_time": decoding_time,
            "alignment_time": alignment_time,
            "encoding_time": de_time
        }

        # Save output
        with open(os.path.join(output_path, f"{filename}.json"), 'w', encoding='utf8') as f:
            json.dump(
                {
                    'segments': ao,
                    'duration': librosa.get_duration(y=signal, sr=16000),
                    'metadata': mtd
                },
                f,
                indent=4,
                ensure_ascii=False
            )

if __name__=="__main__":
    fire.Fire(main)
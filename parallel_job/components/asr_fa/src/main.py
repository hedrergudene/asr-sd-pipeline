# Libraries
import argparse
import sys
import logging as log
import soundfile as sf
import librosa
from pathlib import Path
import os
import re
import time
from typing import List
import json
from num2words import num2words
import torch
import pandas as pd
from faster_whisper import WhisperModel
import whisperx

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

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
    parser.add_argument("--whisper_model_name", type=str, default='large-v2')
    parser.add_argument("--align_model_name", type=str, default='jonatasgrosman/wav2vec2-large-xlsr-53-spanish')
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--vad_threshold", type=float, default=0.5)
    parser.add_argument("--min_speech_duration_ms", type=int, default=200)
    parser.add_argument("--min_silence_duration_ms", type=int, default=500)
    parser.add_argument("--compute_type", type=str, default='float16')
    parser.add_argument("--language_code", type=str, default='es')
    parser.add_argument("--output_path", type=str)
    args, _ = parser.parse_known_args()

    # Device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    global beam_size, vad_threshold, min_speech_duration_ms, min_silence_duration_ms, language_code, output_path
    beam_size = args.beam_size
    vad_threshold = args.vad_threshold
    min_speech_duration_ms = args.min_speech_duration_ms
    min_silence_duration_ms = args.min_silence_duration_ms
    language_code = args.language_code
    output_path = args.output_path

    # Folder structure
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ASR models
    global whisper_model
    whisper_model = WhisperModel(
        args.whisper_model_name,
        device=device,
        compute_type=args.compute_type
    )

    global alignment_model, metadata
    alignment_model, metadata = whisperx.load_align_model(
        language_code = language_code,
        model_name = args.align_model_name,
        device=device
    )


def run(mini_batch):
    for elem in mini_batch:
        # Read file
        pathdir = Path(elem)
        filename, extension = os.path.splitext(str(pathdir).split('/')[-1])
        if os.path.splitext(pathdir)[1] not in ['.wav', '.mp3']:
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
        sf.write(f'./input_audios/{filename}.wav', signal, 16000, 'PCM_24') # save in tmp path as 16kHz, mono
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
        segs = [{'start':x.start, 'end':x.end, 'text':x.text.strip()} for x in segments]
        transcription_time = time.time()-transcription_time
        log.info(f"\t\tTranscription time: {transcription_time}")

        #
        # Number decoding
        #
        log.info(f"Digits decoding:")
        decoding_time = time.time()
        length, ents, seg_dct = 0, [], []
        for x in segs:
            text = x.get('text')
            # Matches that are related to numeric expressions and special symbols
            regex = r'(?:[0-9]+(?:[-.,][0-9]+)*)'
            next_match = re.search(r'(?:[0-9]+(?:[-.,][0-9]+)*)', text)
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
                next_match = re.search(r'(?:[0-9]+(?:[-.,][0-9]+)*)', text)
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
        log.info(f"\tAlignment:")
        alignment_time = time.time()
        result_aligned = whisperx.align(
            seg_dct,
            alignment_model,
            metadata,
            signal,
            device
        )
        alignment_time = time.time() - alignment_time
        log.info(f"\t\tAlignment time: {alignment_time}")

        #
        # Digit (re)encoding
        #
        # Add ABSOLUTE start-end indices tokens to aligned output
        log.info(f"\tDigits (re)encoding:")
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
        for seg in aligned_output:
            shift_seg = 0
            seg['start_idx'], seg['end_idx'] = seg['start_idx']+shift_global, seg['end_idx']+shift_global
            seg['words'] = [{'word': x['word'], 'start': x['start'], 'end': x['end'], 'score': x['score'], 'start_idx': x['start_idx']+shift_global, 'end_idx': x['end_idx']+shift_global} for x in seg['words']]
            ents_seg = [x for x in ents if ((seg['start_idx']<=x['start_idx']) & (seg['end_idx']>=x['end_idx']))]
            if len(ents_seg)>0:
                ents_seg = [x.to_dict() for _, x in pd.DataFrame(ents_seg).sort_values(by='start_idx').iterrows()]
                for ent in ents_seg:
                    # Update entity positions...
                    new_text = ent['pattern']+'.' if ent['end_idx']+1==seg['end_idx'] else ent['pattern']
                    ent = {'text': ent['text'], 'pattern': new_text, 'start_idx': ent['start_idx']+shift_seg, 'end_idx': ent['end_idx']+shift_seg}
                    # ...find all words between those positions and compute stats...
                    target_words_begin = [x for x in seg['words'] if ((x['start_idx']>=ent['start_idx']) & (x['start_idx']<ent['end_idx']))]
                    agg_dct = pd.DataFrame(target_words_begin).agg({'start': 'min', 'end': 'max', 'score': 'mean', 'start_idx': 'min', 'end_idx': 'max'}).to_dict()
                    agg_dct['start_idx'] = int(ent['start_idx'])
                    agg_dct['end_idx'] = ent['start_idx']+len(new_text)
                    agg_dct['word'] = new_text
                    shift_ent = len(new_text)-int(len(' '.join([x['word'] for x in target_words_begin])))
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
                ents = [
                    {
                        'text': x['text'],
                        'pattern': x['pattern'],
                        'start_idx': x['start_idx']+shift_seg,
                        'end_idx': x['end_idx']+shift_seg
                    } for x in ents
                ]
                shift_global += shift_seg
        de_time = time.time() - de_time
        log.info(f"\t\tDigit (re)encoding time: {de_time}")

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
                    'segments': aligned_output,
                    'duration': librosa.get_duration(y=signal, sr=16000),
                    'metadata': mtd
                },
                f,
                ensure_ascii=False
            )
        ## Generate output (filename goes WITHOUT extension, we no longer give a f**k!)
        #objs.append({'filename': filename, 'segments': aligned_output})       

    # Remove audios
    delete_files_in_directory_and_subdirectories('./input_audios')

    return mini_batch
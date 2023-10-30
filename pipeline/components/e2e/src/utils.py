# Libraries
import subprocess
import re
import json
import os
from typing import List, Dict

# Helper method to decode an audio
def preprocess_audio(input_path, output_path, filename):
    """Method to preprocess audios with ffmpeg, using the following configuration:
        * '-acodec': Specifies the audio codec to be used. In this case, it's set to 'pcm_s16le',
                     which stands for 16-bit little-endian PCM (Pulse Code Modulation).
                     This is a standard audio format.
        * '-ac' '1': Sets the number of audio channels to 1, which is mono audio.
        * '-ar' '16000': Sets the audio sample rate to 16 kHz.

    Args:
        input_filepath (str): Folder where audio lies
        output_filepath (str): Folder where audio is to be stored after processing
        filename (str): Name of the file (with extension) you are processing.
    """
    fn, ext = os.path.splitext(filename)
    command = ['ffmpeg', '-i', f"{input_path}/{filename}", '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', f"{output_path}/{fn}.wav"]
    out = subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    if out.returncode!=0:
        raise RuntimeError(f"An error occured during audio preprocessing. Logs are: {out.stderr}")

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
        word_ts = [[float(y.split(' ')[2]), float(y.split(' ')[2])+float(y.split(' ')[3])] for y in word_level_ts] #for diarization purposes
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
    return sg, word_ts

# Helper function to process diarization output
def process_diar_output(diar_output):
    return {fp:[{'start':float(x.split(' ')[0]), 'end': float(x.split(' ')[1]), 'speaker':x.split(' ')[2][-1]} for x in segments] for fp, segments in diar_output.items()}



# Sentence alignment utils
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
        ws, we, wrd = (
            wrd_dict["start"],
            wrd_dict["end"],
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
    words = []

    snt = {
        "speaker": f"Speaker {spk}",
        "start_time": s,
        "end_time": e,
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
                "text": "",
                "words": [],
            }
            words = []
        else:
            snt["end_time"] = wrd_dict["end_time"]
            words.append(wrd_dict)
            snt['words'] = words
        snt["text"] += wrd + " "
        prev_spk = spk
    snt['text'] = snt['text'].strip()
    snts.append(snt)
    return snts
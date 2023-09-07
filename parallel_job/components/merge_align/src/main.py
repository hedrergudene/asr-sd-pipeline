# Libraries
import argparse
import sys
import logging as log
import re
import json
import os
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel

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
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
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
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk
        
    snt['text'] = snt['text'].strip()
    snts.append(snt)
    return snts

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
    parser.add_argument("--max_words_in_sentence", type=int, default=40)
    parser.add_argument("--sentence_ending_punctuations", type=str, default='.?!')
    parser.add_argument("--output_path", type=str)
    args, _ = parser.parse_known_args()

    # Params
    global input_diar_path, max_words_in_sentence, sentence_ending_punctuations, output_path
    input_diar_path = args.input_diar_path
    max_words_in_sentence = args.max_words_in_sentence
    sentence_ending_punctuations = args.sentence_ending_punctuations
    output_path = args.output_path

    # Folder structure
    Path('./input_audios').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Punctuation model
    global punct_model, ending_puncts, model_puncts
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    ending_puncts = ".?!"
    model_puncts = ".,;:!?"


def run(mini_batch):
    for elem in mini_batch: # mini_batch on ASR files
        # Read file
        pathdir = Path(elem)
        filename, _ = os.path.splitext(str(pathdir).split('/')[-1])
        with open(pathdir, 'r', encoding='utf8') as f:
            asr_dct = json.load(f)
        asr_dct = [{'start':w['start'],'end':w['end'],'text':w['word']} for s in asr_dct for w in s['words']]
        with open(os.path.join(input_diar_path, f"{filename}.json"), 'r', encoding='utf8') as f:
            diar_dct = json.load(f)
        diar_dct = [[s['start'], s['end'], s['speaker']] for s in diar_dct]
        # Get labels for each piece of text from ASR
        wsm = get_words_speaker_mapping(asr_dct, diar_dct)
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list)
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
        wsm_final = get_realigned_ws_mapping_with_punctuation(wsm, max_words_in_sentence, sentence_ending_punctuations)
        ssm = get_sentences_speaker_mapping(wsm_final, diar_dct)        

        # Save output
        with open(os.path.join(output_path, f"{filename}.json"), 'w', encoding='utf8') as f:
            json.dump(ssm, f, ensure_ascii=False)

    return mini_batch
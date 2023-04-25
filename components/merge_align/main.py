# Requierments
import logging as log
import re
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import get_file, get_words_speaker_mapping, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping
from deepmultilingualpunctuation import PunctuationModel
import fire


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
    input_asr_path,
    input_diarizer_path,
    max_words_in_sentence,
    sentence_ending_punctuations,
    output_path
):
    # Create output paths
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path('./input_asr').mkdir(parents=True, exist_ok=True)
    Path('./input_diarizer').mkdir(parents=True, exist_ok=True)
    Path('./output_transcriptions').mkdir(parents=True, exist_ok=True)
    # Fetch files
    os.system(f"unrar e {get_file(input_asr_path)} ./input_asr")
    os.system(f"unrar e {get_file(input_diarizer_path)} ./input_diarizer")
    # Get list of filenames
    filenames = [x.split('.')[0] for x in os.listdir('./input_asr')]
    # PunctuationModel and metadata
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # Loop
    for filename in filenames:
        asr_df = pd.DataFrame(f"./input_asr/{filename}.csv")
        asr_dct = [{'start':s,'end':e,'text':t} for (s,e,t) in asr_df.itertuples(index=False)]
        diar_df = pd.DataFrame(f"./input_diarizer/{filename}.csv")
        diar_dct = [[s,e,sp] for (s,e,sp) in diar_df.itertuples(index=False)]
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
        # Save
        pd.DataFrame(ssm).to_csv(f"./output_transcriptions/{filename}.csv")
    # Output
    os.system(f"rar a {os.path.join(output_path,'transcripts.rar')} ./output_transcriptions")


if __name__=="__main__":
    fire.Fire(main)
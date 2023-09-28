# Requierments
import logging as log
import re
import json
import os
import sys
from pathlib import Path
import time
from src.utils import get_words_speaker_mapping, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping
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
    # Get list of filenames
    filenames = [x.split('/')[-1].split('.')[0] for x in os.listdir(input_asr_path)]
    # PunctuationModel and metadata
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    ending_puncts = sentence_ending_punctuations
    model_puncts = ".,;:!?"

    # Loop
    for filename in filenames:
        with open(os.path.join(input_asr_path, f"{filename}.json"), 'r', encoding='utf8') as f:
            asr_dct = json.load(f)
        asr_input = [w for s in asr_dct['segments'] for w in s['words']]
        with open(os.path.join(input_diarizer_path, f"{filename}.json"), 'r', encoding='utf8') as f:
            diar_dct = json.load(f)
        diar_input = [[s['start'], s['end'], s['speaker']] for s in diar_dct['segments']]
        # If file contains no segments, jump to the next one generating dummy metadata
        if len(asr_dct['segments'])==0:
            log.info(f"Audio {filename} does not contain segments. Dumping dummy file and skipping:")
            # Save output
            with open(
                os.path.join(
                    output_path,
                    f"{filename}.json"
                ),
                'w',
                encoding='utf8'
            ) as f:
                json.dump(
                    {
                    'unique_id': filename,
                    'duration': asr_dct['duration'],
                    'processing_time': {
                        **asr_dct['metadata'],
                        **diar_dct['metadata'],
                        **{
                            'sentence_mapping_time': 0
                        }
                    },
                    'segments': []
                },
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            continue
        # Get labels for each piece of text from ASR
        sm_time = time.time()
        wsm = get_words_speaker_mapping(asr_input, diar_input)
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
        ssm = get_sentences_speaker_mapping(wsm_final, diar_input)
        sm_time = time.time() - sm_time
        log.info(f"\tSentence-mapping time: {sm_time}")

        # Save output
        with open(
            os.path.join(
                output_path,
                f"{filename}.json"
            ),
            'w',
            encoding='utf8'
        ) as f:
            json.dump(
                {
                'unique_id': filename,
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


if __name__=="__main__":
    fire.Fire(main)
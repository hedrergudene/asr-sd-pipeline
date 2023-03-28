# Requierments
import logging as log
import re
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
        

# Helper function to combine consecutive rows from the same speaker
def merge_labels(df:pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): Table containing start, end, text and label of each segment

    Returns:
        pd.DataFrame: Table containing start, end, text and label of each merged segment
    """
    df_list = []
    start, end, text, label = [], [], [], ''
    for idx, (s, e, t, l) in enumerate(df.itertuples(index=False)):
        # Cold start
        if idx==0:
            start.append(s)
            end.append(e)
            text.append(t)
            label = l
            continue
        # If speaker does NOT change
        if label==l:
            start.append(s)
            end.append(e)
            text.append(t)
            label = l
            if idx+1==len(df):
                df_list.append({'start':min(start), 'end':max(end), 'text': ' '.join(text), 'label': l})
        # If speaker changes
        else:
            df_list.append({'start':min(start), 'end':max(end), 'text': ' '.join(text), 'label': label})
            start, end, text, label = [s], [e], [t], l
            if idx+1==len(df):
                df_list.append({'start':min(start), 'end':max(end), 'text': ' '.join(text), 'label': label})
    return pd.DataFrame(df_list)


# Helper function to get rid of overlaps
def remove_overlaps(df:pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): Table containing start, end, text and label of each
                           merged segment.

    Returns:
        pd.DataFrame: Table containing start, end, text and label without overlaps
                      surrounded by segmentes of th same speaker.
    """
    df_copy = df.copy()
    labels = df_copy['label'].values
    for idx, label in enumerate(labels):
        if label=='overlap':
            # Fetch next_speaker
            try:
                next_speaker = next(x for x in labels[idx:] if x !='overlap')
            except StopIteration:
                next_speaker = None
            # Fetch prior_speaker
            try:
                previous_speaker = next(x for x in labels[:idx][::-1] if x !='overlap')
            except StopIteration:
                previous_speaker = None
            # Check logic for speakers
            if previous_speaker==next_speaker:
                new_speaker = next_speaker
            elif next_speaker is None:
                new_speaker = previous_speaker
            elif previous_speaker is None:
                new_speaker = next_speaker
            else:
                new_speaker = label
            df_copy.loc[idx,'label'] = new_speaker
    return df_copy


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_asr_path,
    input_diarizer_path,
    output_path
):
    # Create output paths
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path('./input_asr').parent.mkdir(parents=True, exist_ok=True)
    Path('./input_diarizer').parent.mkdir(parents=True, exist_ok=True)
    Path('./output_transcriptions').parent.mkdir(parents=True, exist_ok=True)
    # Fetch files
    os.system(f"unrar e {get_file(input_asr_path)} ./input_asr")
    os.system(f"unrar e {get_file(input_diarizer_path)} ./input_diarizer")
    # Get list of filenames
    filenames = [x.split('.')[0] for x in os.listdir('./input_asr')]

    # Loop
    for filename in filenames:
        asr_df = pd.DataFrame(f"./input_asr/{filename}.csv", index_col=[0])
        diar_df = pd.DataFrame(f"./input_diarizer/{filename}.csv", index_col=[0])
        # Get labels for each piece of text from ASR
        labels = []
        for (start, end, _) in asr_df.itertuples(index=False):
            mid = (start+end)*.5
            # If midpoint lies within a predefined segment
            is_diar = ((mid>=diar_df['start']) & (mid<=diar_df['end']))
            if any(is_diar):
                labels.append(diar_df.loc[is_diar[is_diar].index[0], 'speaker'])
                continue
            # If it does not lie within a segment nor belongs to an overlap interval,
            # find closest interval
            dist_diar = np.min(np.stack([np.abs(mid-diar_df['start'].values), np.abs(mid-diar_df['end'].values)], axis=1), axis=1)
            labels.append(diar_df.loc[np.argmin(dist_diar), 'speaker'])
        asr_df['label'] = labels

        # Merge concatenated labels
        asr_merge_df = merge_labels(asr_df)
        # Greedy solution: Assign overlaps to next speaker; if it's not possible, assign to prior
        asr_merge_df = remove_overlaps(asr_merge_df)
        # Merge concatenated labels (again)
        asr_merge_df = merge_labels(asr_merge_df)
        # Save
        asr_merge_df.to_csv(f"./output_transcriptions/{filename}.csv")
    # Output
    os.system(f"rar a {os.path.join(output_path,'transcripts.rar')} ./output_transcriptions")


if __name__=="__main__":
    fire.Fire(main)
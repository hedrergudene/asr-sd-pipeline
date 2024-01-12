# Requierments
import logging as log
import re
import os
import sys
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import fire

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Helper method to remove blobs
def delete_blob(blob_service_client: BlobServiceClient, container_name: str, blob_name: str):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.delete_blob()

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    storage_id,
    container_name,
    blob_filepath
):
    # Check if given credential can get token successfully
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    # Create a blob client using the local file name as the name for the blob
    account_url = f"https://{storage_id}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    # Loop
    regex_fn = lambda pattern,text : len(re.findall(pattern, text))>0
    for elem in os.listdir(input_path):
        if ((regex_fn('\.wav\.pgp', elem)) | (regex_fn('_prep', elem))| (regex_fn('_asr', elem))| (regex_fn('_nfa', elem))| (regex_fn('_diar', elem))):
            delete_blob(blob_service_client, container_name, os.path.join(blob_filepath,elem))
        else:
            continue

if __name__=="__main__":
    fire.Fire(main)
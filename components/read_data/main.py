# Requierments
import logging as log
import re
import json
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

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    storage_id,
    container_id,
    regex_pattern,
    output_path
):
    # Create output paths
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Define variables
    account_url = f"https://{storage_id}.blob.core.windows.net"
    # Check if given credential can get token successfully
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    # Create a blob client using the local file name as the name for the blob
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    # Get files
    container_client = blob_service_client.get_container_client(container=container_id)
    blob_list = container_client.list_blobs()
    # Regular expression to evaluate blob names
    regex_fn = lambda text: re.findall(regex_pattern, text)
    # Iterate through blobs
    annot_list = []
    for blob in blob_list:
        # If name does not match criteria
        if len(regex_fn(blob.name))==0:
            continue
        else:
            filename, extension = os.path.splitext(blob.name)
            log.info(f"Processing filename {filename}:")
            #elem = eval(container_client.download_blob(blob.name).readall())
            annot_list.append(f"https://{storage_id}.blob.core.windows.net/{container_id}/{filename}{extension}")
    # If no files were found, stop the script
    if len(annot_list)==0:
        log.error("No annotations following the introduced criteria were found.")
        return RuntimeError("No annotations following the introduced criteria were found.")
    # If files were found, generate output
    pd.DataFrame({'paths':annot_list}).to_csv(os.path.join(output_path, 'output.csv'))


if __name__=="__main__":
    fire.Fire(main)
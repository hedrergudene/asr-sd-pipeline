# Libraries
import yaml
import sys
import os
import logging as log
from azure.ai.ml import MLClient, load_component
from azure.ai.ml.entities import BatchEndpoint, PipelineComponentBatchDeployment
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
    config_path:str='./config/online_endpoint.yaml'
):

    # Get credential token
    log.info("Get credential token:")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        log.error(f"Something went wrong regarding authentication. Returned error is: {ex.message}")
        return (f"Something went wrong regarding authentication. Returned error is: {ex.message}", 500)
    
    # Fetch configuration file
    log.info("Fetch configuration file:")
    with open(config_path) as file:
        config_dct = yaml.load(file, Loader=yaml.FullLoader)
    # Get a handle to workspace
    log.info("Set up ML Client:")
    ml_client = MLClient(
        credential=credential,
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name'],
    )

    # Define the endpoint
    log.info("Define batch endpoint:")
    endpoint = BatchEndpoint(
        name=config_dct['endpoint']['name'],
        description=config_dct['endpoint']['description']
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    # Load registered component
    log.info("Load registered component:")
    stt_batch = load_component(client=ml_client, name="stt", version="1")

    # Deploy pipeline component
    log.info("Deploy pipeline component:")
    deployment = PipelineComponentBatchDeployment(
        name=config_dct['deployment']['name'],
        description=config_dct['deployment']['description'],
        endpoint_name=endpoint.name,
        component=stt_batch,
        settings={"continue_on_step_failure": False, "default_compute": config_dct['deployment']['default_compute']},
    )


if __name__=="__main__":
    fire.Fire(main)
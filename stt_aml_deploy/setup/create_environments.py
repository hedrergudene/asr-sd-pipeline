# Libraries
import yaml
import sys
import os
import re
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
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

    # Build environments for AML pipeline
    for comp_name in os.listdir('../src/components'):
        log.info(f"Building environment {comp_name}:")
        env_docker_context = Environment(
            build=BuildContext(path=f"../src/components/{comp_name}/docker"),
            name=f"{comp_name}_env",
            description=f"Environment for {comp_name} component of speech to text solution.",
        )
        ml_client.environments.create_or_update(env_docker_context)

    # Build environments for endpoint
    log.info(f"Building environment for speech to text pipeline endpoint:")
    env_docker_context = Environment(
        build=BuildContext(path=f"../docker"),
        name=f"stt_env",
        description=f"Environment for speech to text pipeline endpoint.",
    )
    ml_client.environments.create_or_update(env_docker_context)

if __name__=="__main__":
    fire.Fire(main)
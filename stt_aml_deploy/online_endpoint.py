# Libraries
import yaml
import sys
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    BuildContext,
    CodeConfiguration
)
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
    log.info(f"Fetch credential token:")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        log.error(f"Something went wrong regarding authentication. Returned error is: {ex.message}")
        return (f"Something went wrong regarding authentication. Returned error is: {ex.message}", 500)
    
    # Fetch configuration file
    with open(config_path, 'r') as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader) 

    # Azure workspace config
    log.info(f"Get MLClient:")
    ml_client = MLClient(
        credential=credential,
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name']
    )

    # Create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=config_dct['endpoint']['name'],
        description=config_dct['endpoint']['description'],
        auth_mode=config_dct['endpoint']['auth_mode']
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint)

    # Define environment
    env = ml_client.environments.get(name="stt_env", version="1")

    # Deployment configuration
    dpl = ManagedOnlineDeployment(
        name=config_dct['deployment']['name'],
        endpoint_name=config_dct['endpoint']['name'],
        environment=env,
        code_configuration=CodeConfiguration(
            code="./src",
            scoring_script="score.py"
        ),
        instance_type=config_dct['deployment']['instance_type'],
        instance_count=config_dct['deployment']['instance_count']
    )
    ml_client.online_deployments.begin_create_or_update(dpl)

    # Switch traffic
    endpoint.traffic={
            'blue': config_dct['endpoint']['traffic']['blue'],
            'green': config_dct['endpoint']['traffic']['green']
        }
    ml_client.online_endpoints.begin_create_or_update(endpoint)

    # For a detailed overview of autoscale settings, check:
    # https://learn.microsoft.com/es-es/azure/machine-learning/how-to-autoscale-endpoints?view=azureml-api-2&tabs=python

    # For a detailed overview of mirrored deployments, check:
    # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2&tabs=azure-cli
    
if __name__=="__main__":
    fire.Fire(main)
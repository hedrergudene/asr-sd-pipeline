# Libraries
import yaml
import sys
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
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
    config_path:str='./config/asr_msdd_inference_pipeline.yaml'
):

    # Get credential token
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        log.error(f"Something went wrong regarding authentication. Returned error is: {ex.message}")
        return (f"Something went wrong regarding authentication. Returned error is: {ex.message}", 500)
    
    # Fetch configuration file
    with open(config_path) as file:
        config_dct = yaml.load(file, Loader=yaml.FullLoader)
    # Get a handle to workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name'],
    )
    
    # Define the gpu cluster computations if it is not found
    try:
        cpu_cluster = ml_client.compute.get(config_dct['azure']['computing']['cpu_cluster_aml_id'])
    except Exception as ex:
        log.error(f"CPU cluster was not found. Introduced parameter is {config_dct['azure']['computing']['cpu_cluster_aml_id']}. Returned error is: {ex.message}")
    try:
        gpu_cluster = ml_client.compute.get(config_dct['azure']['computing']['gpu_cluster_aml_id'])
    except Exception as ex:
        log.error(f"GPU cluster was not found. Introduced parameter is {config_dct['azure']['computing']['gpu_cluster_aml_id']}. Returned error is: {ex.message}")


    # Fetch components
    read_data_comp = load_component(source="./components/read_data/read_data.yaml")
    asr_inference_comp = load_component(source="./components/asr_inference/asr_inference.yaml")
    diarize_inference_comp = load_component(source=f"./components/diarize_inference/diarize_inference.yaml")
    merge_asr_diar_comp = load_component(source=f"./components/merge_asr_diar/merge_asr_diar.yaml")
    
    # Define a pipeline containing the previous nodes
    @pipeline(
        default_compute=config_dct['azure']['computing']['cpu_cluster_aml_id'],
    )
    def asr_msdd_inference_pipeline():
        """Multi-speaker speech recognition pipeline."""
        # Read 
        read_data_node = read_data_comp(
            storage_id=config_dct['data']['storage_id'],
            container_id=config_dct['data']['container_id'],
            regex_pattern=config_dct['data']['regex_pattern']
        )

        # ASR
        asr_inference_node = asr_inference_comp(
            input_path=read_data_node.outputs.output_path,
            whisper_model_name=config_dct['inference']['whisper_model_name'],
            vad_threshold=config_dct['inference']['vad_threshold'],
            no_speech_threshold=config_dct['inference']['no_speech_threshold'],
            language_code=config_dct['inference']['language_code']
        )
        asr_inference_node.compute = config_dct['azure']['computing']['gpu_cluster_aml_id']

        # MSDD
        diarize_inference_node = diarize_inference_comp(
            input_path=read_data_node.outputs.output_path,
            event_type=config_dct['inference']['event_type']
        )
        diarize_inference_node.compute = config_dct['azure']['computing']['gpu_cluster_aml_id']

        # Merge
        merge_asr_diar_node= merge_asr_diar_comp(
            input_asr_path=asr_inference_node.outputs.output_path,
            input_diarizer_path=diarize_inference_node.outputs.output_path
        )

    
    # Create a pipeline
    pipeline_job = asr_msdd_inference_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=config_dct['azure']['experiment_name']
    )

if __name__=="__main__":
    fire.Fire(main)
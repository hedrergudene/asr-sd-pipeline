# Libraries
import yaml
import sys
import os
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.constants import AssetTypes, InputOutputModes
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
    config_path:str='./config/pipeline.yaml'
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

    # Register environments
    log.info("Check environments availability:")
    envs = [x.name for x in ml_client.environments.list()]
    env2version = {}
    for x in os.listdir('./components'):
        env_name = f"{x}_env"
        if env_name not in envs:
            log.info(f"Environment for component {x} not found. Creating...")
            ml_client.environments.create_or_update(
                Environment(
                    build=BuildContext(path=f"./components/{x}/docker"),
                    name=env_name
                )
            )
            log.info(f"Environment for component {x} created.")
            env2version[env_name] = "1"
        else:
            env2version[env_name] = str(max([int(x.version) for x in ml_client.environments.list(name=env_name)]))
            log.info(f"Environment for component {x} was found. Latest version is {env2version[env_name]}.")
            if int(env2version[env_name])>1:
                log.info(f"Updating environment for component {x} to latest version:")
                with open(f"./components/{x}/{x}.yaml") as fenv:
                    env_dct = yaml.load(fenv, Loader=yaml.FullLoader)
                env_dct['environment']['image'] = f"{env_name}:{env2version[env_name]}"
                with open(f"./components/{x}/{x}.yaml", 'w') as fenv:
                    yaml.dump(env_dct, fenv)
                

    # Set the input and output URI paths for the data.
    input_audio_data = Input(
        path=config_dct['azure']['data_filepath'],
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.RO_MOUNT #Alternative, DOWNLOAD
    )
    
    # Fetch components
    asr_comp = load_component(source="./components/asr/asr.yaml")
    fa_comp = load_component(source="./components/fa/fa.yaml")
    diar_comp = load_component(source=f"./components/diar/diar.yaml")
    merge_align_comp = load_component(source=f"./components/merge_align/merge_align.yaml")
    
    # Define a pipeline containing the previous nodes
    @pipeline(
        default_compute=config_dct['azure']['computing']['gpu_cluster_aml_id'],
    )
    def asr_msdd_inference_pipeline(
        input_data:Input
    ):
        """Multi-speaker speech recognition pipeline."""

        # ASR
        asr_node = asr_comp(
            input_path=input_data,
            whisper_model_name=config_dct['asr']['whisper_model_name'],
            num_workers=config_dct['asr']['num_workers'],
            beam_size=config_dct['asr']['beam_size'],
            vad_threshold=config_dct['asr']['vad_threshold'],
            min_speech_duration_ms=config_dct['asr']['min_speech_duration_ms'],
            min_silence_duration_ms=config_dct['asr']['min_silence_duration_ms'],
            word_level_timestamps=config_dct['asr']['word_level_timestamps'],
            compute_type=config_dct['asr']['compute_type'],
            language_code=config_dct['asr']['language_code']
        )

        # FA
        fa_node = fa_comp(
            input_path=input_data,
            input_asr_path=asr_node.outputs.output_path,
            align_model_name=config_dct['asr']['align_model_name'],
            language_code=config_dct['asr']['language_code']
        )

        # MSDD
        diar_node = diar_comp(
            input_path=input_data,
            input_asr_path=fa_node.outputs.output_path,
            event_type=config_dct['diarization']['event_type'],
            max_num_speakers=config_dct['diarization']['max_num_speakers']
        )

        # Merge
        merge_align_node= merge_align_comp(
            input_asr_path=fa_node.outputs.output_path,
            input_diarizer_path=diar_node.outputs.output_path,
            max_words_in_sentence=config_dct['align']['max_words_in_sentence'],
            sentence_ending_punctuations=config_dct['align']['sentence_ending_punctuations']
        )

        return {
            "output": merge_align_node.outputs.output_path
        }

    
    # Create a pipeline
    pipeline_job = asr_msdd_inference_pipeline(input_audio_data)
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=config_dct['azure']['experiment_name']
    )

if __name__=="__main__":
    fire.Fire(main)

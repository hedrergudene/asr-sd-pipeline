# Libraries
import yaml
import sys
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.constants import AssetTypes, InputOutputModes
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
    config_path:str='./config/parallel_job.yaml'
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
        subscription_id=config_dct['aml']['subscription_id'],
        resource_group_name=config_dct['aml']['resource_group'],
        workspace_name=config_dct['aml']['workspace_name'],
    )
        

    # Set the input and output URI paths for the data.
    input_dts = Input(
        path=config_dct['blob']['input_path'],
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.RO_MOUNT #Alternative, DOWNLOAD
    )

    # Load registered component
    stt_batch = load_component(client=ml_client, name="stt", version="1")


    # Create a pipeline
    pipeline_job = stt_batch(
        input_dts = input_dts,
        output_dts = config_dct['blob']['output_path'],
        storage_account_name = config_dct['blob']['storage_id'],
        container_name = config_dct['blob']['container_name'],
        blob_filepath = config_dct['blob']['blob_filepath'],
        aml_cpu_cluster = config_dct['aml']['computing']['cpu_cluster'],
        aml_t4_cluster = config_dct['aml']['computing']['gpu_cluster_t4'],
        aml_a100_cluster = config_dct['aml']['computing']['gpu_cluster_a100'],
        keyvault_name = config_dct['keyvault']['name'],
        secret_tenant_sp = config_dct['keyvault']['secret_tenant_sp'],
        secret_client_sp = config_dct['keyvault']['secret_client_sp'],
        secret_sp = config_dct['keyvault']['secret_sp'],
        pk_secret = config_dct['keyvault']['pk_secret'],
        pk_pass_secret = config_dct['keyvault']['pk_pass_secret'],
        pubk_secret = config_dct['keyvault']['pubk_secret'],
        cosmosdb_name = config_dct['cosmosdb']['name'],
        cosmosdb_collection = config_dct['cosmosdb']['collection'],
        cosmosdb_cs_secret = config_dct['cosmosdb']['cs_secret'],
        vad_threshold = config_dct['preprocessing']['vad_threshold'],
        min_speech_duration_ms = config_dct['preprocessing']['min_speech_duration_ms'],
        min_silence_duration_ms = config_dct['preprocessing']['min_silence_duration_ms'],
        demucs_model = config_dct['preprocessing']['demucs_model'],
        asr_model_name = config_dct['asr']['model_name'],
        asr_num_workers = config_dct['asr']['num_workers'],
        asr_beam_size = config_dct['asr']['beam_size'],
        word_level_timestamps = config_dct['asr']['word_level_timestamps'],
        condition_on_previous_text = config_dct['asr']['condition_on_previous_text'],
        asr_compute_type = config_dct['asr']['compute_type'],
        asr_language_code = config_dct['asr']['language_code'],
        nfa_model_name = config_dct['fa']['model_name'],
        nfa_batch_size = config_dct['fa']['batch_size'],
        diar_event_type = config_dct['diarization']['event_type'],
        diar_max_num_speakers = config_dct['diarization']['max_num_speakers'],
        diar_min_window_length = config_dct['diarization']['min_window_length'],
        diar_overlap_threshold = config_dct['diarization']['overlap_threshold'],
        ma_ner_chunk_size = config_dct['align']['ner_chunk_size'],
        ma_ner_stride = config_dct['align']['ner_stride'],
        ma_max_words_in_sentence = config_dct['align']['max_words_in_sentence']
    )
    # Run job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=config_dct['aml']['project_name']
    )


if __name__=="__main__":
    fire.Fire(main)
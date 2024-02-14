# Libraries
import yaml
import sys
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline
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

# Main method. Fire automatically align method arguments with parse commands from console
def main(
    config_path='../config/parallel_job.yaml'
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


    # Fetch components
    prep_comp = load_component(client=ml_client, name="prep", version="1")
    asr_comp = load_component(client=ml_client, name="asr", version="1")
    nfa_comp = load_component(client=ml_client, name="nfa", version="1")
    diar_comp = load_component(client=ml_client, name="diar", version="1")
    ma_comp = load_component(client=ml_client, name="merge_align", version="1")
    rsttd_comp = load_component(client=ml_client, name="remove_stt_data", version="1")

    #
    # Create pipeline
    #


    @pipeline()
    def stt(
        input_dts:Input(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RO_MOUNT),
        output_dts:Input(type='string'),
        storage_account_name:Input(type="string"),
        container_name:Input(type="string"),
        blob_filepath:Input(type="string"),
        aml_cpu_cluster:Input(type="string"),
        aml_t4_cluster:Input(type="string"),
        aml_a100_cluster:Input(type="string"),
        keyvault_name:Input(type="string"),
        secret_tenant_sp:Input(type="string"),
        secret_client_sp:Input(type="string"),
        secret_sp:Input(type="string"),
        pk_secret:Input(type="string"),
        pk_pass_secret:Input(type="string"),
        pubk_secret:Input(type="string"),
        cosmosdb_name:Input(type="string"),
        cosmosdb_collection:Input(type="string"),
        cosmosdb_cs_secret:Input(type="string"),
        vad_threshold:Input(type="number", default=config_dct['preprocessing']['vad_threshold'], optional=True),
        min_speech_duration_ms:Input(type="integer", default=config_dct['preprocessing']['min_speech_duration_ms'], optional=True),
        min_silence_duration_ms:Input(type="integer", default=config_dct['preprocessing']['min_silence_duration_ms'], optional=True),
        demucs_model:Input(type="string", default=config_dct['preprocessing']['demucs_model'], optional=True),
        asr_model_name:Input(type="string", default=config_dct['asr']['model_name'], optional=True),
        asr_num_workers:Input(type="integer", default=config_dct['asr']['num_workers'], optional=True),
        asr_beam_size:Input(type="integer", default=config_dct['asr']['beam_size'], optional=True),
        word_level_timestamps:Input(type="boolean", default=config_dct['asr']['word_level_timestamps'], optional=True),
        condition_on_previous_text:Input(type="boolean", default=config_dct['asr']['condition_on_previous_text'], optional=True),
        asr_compute_type:Input(type="string", default=config_dct['asr']['compute_type'], optional=True),
        asr_language_code:Input(type="string", default=config_dct['asr']['language_code'], optional=True),
        nfa_model_name:Input(type="string", default=config_dct['fa']['model_name'], optional=True),
        nfa_batch_size:Input(type="integer", default=config_dct['fa']['batch_size'], optional=True),
        diar_event_type:Input(type="string", default=config_dct['diarization']['event_type'], optional=True),
        diar_max_num_speakers:Input(type="integer", default=config_dct['diarization']['max_num_speakers'], optional=True),
        diar_min_window_length:Input(type="number", default=config_dct['diarization']['min_window_length'], optional=True),
        diar_overlap_threshold:Input(type="number", default=config_dct['diarization']['overlap_threshold'], optional=True),
        ma_ner_chunk_size:Input(type="integer", default=config_dct['align']['ner_chunk_size'], optional=True),
        ma_ner_stride:Input(type="integer", default=config_dct['align']['ner_stride'], optional=True),
        ma_max_words_in_sentence:Input(type="integer", default=config_dct['align']['max_words_in_sentence'], optional=True)
    ):

        # Preprocessing
        prep_node = prep_comp(
            input_dts=input_dts,
            output_dts=output_dts,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            cosmosdb_name=cosmosdb_name,
            cosmosdb_collection=cosmosdb_collection,
            cosmosdb_cs_secret=cosmosdb_cs_secret,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            demucs_model=demucs_model,
            compute_cluster=aml_t4_cluster
        )

        # ASR
        asr_node = asr_comp(
            input_dts=prep_node.outputs.output_dts,
            output_dts=output_dts,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            model_name=asr_model_name,
            num_workers=asr_num_workers,
            beam_size=asr_beam_size,
            word_level_timestamps=word_level_timestamps,
            condition_on_previous_text=condition_on_previous_text,
            compute_type=asr_compute_type,
            language_code=asr_language_code,
            compute_cluster=aml_t4_cluster
        )

        # NFA
        nfa_node = nfa_comp(
            input_dts=prep_node.outputs.output_dts,
            input_asr=asr_node.outputs.output_dts,
            output_dts=output_dts,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            model_name=nfa_model_name,
            batch_size=nfa_batch_size,
            compute_cluster=aml_t4_cluster
        )

        # Diarization
        diar_node = diar_comp(
            input_dts=prep_node.outputs.output_dts,
            input_asr=nfa_node.outputs.output_dts,
            output_dts=output_dts,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            event_type=diar_event_type,
            max_num_speakers=diar_max_num_speakers,
            min_window_length=diar_min_window_length,
            overlap_threshold=diar_overlap_threshold,
            compute_cluster=aml_a100_cluster
        )

        # Merge&Align
        ma_node = ma_comp(
            input_asr = nfa_node.outputs.output_dts,
            input_diar = diar_node.outputs.output_dts,
            output_dts=output_dts,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            cosmosdb_name=cosmosdb_name,
            cosmosdb_collection=cosmosdb_collection,
            cosmosdb_cs_secret=cosmosdb_cs_secret,
            ner_chunk_size=ma_ner_chunk_size,
            ner_stride=ma_ner_stride,
            max_words_in_sentence=ma_max_words_in_sentence,
            compute_cluster=aml_t4_cluster
        )

        # Remove STT data
        rsttd_node = rsttd_comp(
            input_path = ma_node.outputs.output_dts,
            storage_id = storage_account_name,
            container_name = container_name,
            blob_filepath = blob_filepath
        )
        rsttd_node.compute = aml_cpu_cluster


    # Create a pipeline
    pipeline_job = stt()

    # Component register
    ml_client.components.create_or_update(pipeline_job.component, version="1")


if __name__=="__main__":
    fire.Fire(main)
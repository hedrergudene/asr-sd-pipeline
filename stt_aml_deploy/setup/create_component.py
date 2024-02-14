# Libraries
import yaml
import sys
import os
import re
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import RetrySettings
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.parallel import parallel_run_function, RunFunction
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
    rsttd_comp = load_component(source="../components/remove_stt_data/remove_stt_data.yaml")


    #
    # Declare Parallel task to perform preprocessing
    # For detailed info, check: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=python
    #
    prep_component = parallel_run_function(
        name="pPrep",
        display_name="Parallel preprocessing",
        description="Parallel component to perform audio preprocessing",
        inputs=dict(
            input_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be preprocessed"),
            keyvault_name=Input(type="string"),
            secret_tenant_sp=Input(type="string"),
            secret_client_sp=Input(type="string"),
            secret_sp=Input(type="string"),
            pk_secret=Input(type="string"),
            pk_pass_secret=Input(type="string"),
            pubk_secret=Input(type="string"),
            cosmosdb_name=Input(type="string"),
            cosmosdb_collection=Input(type="string"),
            cosmosdb_cs_secret=Input(type="string"),
            vad_threshold=Input(type="number", default=config_dct['preprocessing']['vad_threshold'], optional=True),
            min_speech_duration_ms=Input(type="integer", default=config_dct['preprocessing']['min_speech_duration_ms'], optional=True),
            min_silence_duration_ms=Input(type="integer", default=config_dct['preprocessing']['min_silence_duration_ms'], optional=True),
            demucs_model=Input(type="string", default=config_dct['preprocessing']['demucs_model'], optional=True)
        ),
        outputs=dict(
            output_prep_path=Output(type=AssetTypes.URI_FOLDER)
        ),
        input_data="${{inputs.input_path}}",
        instance_count=config_dct['job']['instance_count'],
        max_concurrency_per_instance=config_dct['job']['max_concurrency_per_instance'],
        mini_batch_size=config_dct['job']['mini_batch_size'],
        mini_batch_error_threshold=config_dct['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=config_dct['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=config_dct['job']['max_retries'],
            timeout=config_dct['job']['timeout']
        ),
        task=RunFunction(
            code="../components/prep/src",
            entry_script="main.py",
            environment=ml_client.environments.get(name="prep_env", version="1"),
            program_arguments="--input_path ${{inputs.input_path}} "
                                "--keyvault_name ${{inputs.keyvault_name}} "
                                "--secret_tenant_sp ${{inputs.secret_tenant_sp}} "
                                "--secret_client_sp ${{inputs.secret_client_sp}} "
                                "--secret_sp ${{inputs.secret_sp}} "
                                "--pk_secret ${{inputs.pk_secret}} "
                                "--pk_pass_secret ${{inputs.pk_pass_secret}} "
                                "--pubk_secret ${{inputs.pubk_secret}} "
                                "--cosmosdb_name ${{inputs.cosmosdb_name}} "
                                "--cosmosdb_collection ${{inputs.cosmosdb_collection}} "
                                "--cosmosdb_cs_secret ${{inputs.cosmosdb_cs_secret}} "
                                "$[[--vad_threshold ${{inputs.vad_threshold}}]] "
                                "$[[--min_speech_duration_ms ${{inputs.min_speech_duration_ms}}]] "
                                "$[[--min_silence_duration_ms ${{inputs.min_silence_duration_ms}}]] "
                                "$[[--demucs_model ${{inputs.demucs_model}}]] "
                                "--output_prep_path ${{outputs.output_prep_path}} "
                                f"--allowed_failed_percent {config_dct['job']['allowed_failed_percent']} "
                                f"--progress_update_timeout {config_dct['job']['progress_update_timeout']} "
                                f"--task_overhead_timeout {config_dct['job']['task_overhead_timeout']} "
                                f"--first_task_creation_timeout {config_dct['job']['first_task_creation_timeout']} "
                                f"--resource_monitor_interval {config_dct['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )

    #
    # Declare Parallel task to perform ASR
    # For detailed info, check: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=python
    #
    asr_component = parallel_run_function(
        name="pASR",
        display_name="Parallel ASR",
        description="Parallel component to perform ASR on a large amount of audios",
        inputs=dict(
            input_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be transcribed and metadata attached to those audios."),
            keyvault_name=Input(type="string"),
            secret_tenant_sp=Input(type="string"),
            secret_client_sp=Input(type="string"),
            secret_sp=Input(type="string"),
            pk_secret=Input(type="string"),
            pk_pass_secret=Input(type="string"),
            pubk_secret=Input(type="string"),
            whisper_model_name=Input(type="string", default=config_dct['asr']['model_name'], optional=True),
            num_workers=Input(type="integer", default=config_dct['asr']['num_workers'], optional=True),
            beam_size=Input(type="integer", default=config_dct['asr']['beam_size'], optional=True),
            word_level_timestamps=Input(type="boolean", default=config_dct['asr']['word_level_timestamps'], optional=True),
            condition_on_previous_text=Input(type="boolean", default=config_dct['asr']['condition_on_previous_text'], optional=True),
            compute_type=Input(type="string", default=config_dct['asr']['compute_type'], optional=True),
            language_code=Input(type="string", default=config_dct['asr']['language_code'], optional=True)
        ),
        outputs=dict(output_asr_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_path}}",
        instance_count=config_dct['job']['instance_count'],
        max_concurrency_per_instance=config_dct['job']['max_concurrency_per_instance'],
        mini_batch_size=config_dct['job']['mini_batch_size'],
        mini_batch_error_threshold=config_dct['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=config_dct['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=config_dct['job']['max_retries'],
            timeout=config_dct['job']['timeout']
        ), 
        task=RunFunction(
            code="../components/asr/src",
            entry_script="main.py",
            environment=ml_client.environments.get(name="asr_env", version="1"),
            program_arguments="--input_path ${{inputs.input_path}} "
                                "--keyvault_name ${{inputs.keyvault_name}} "
                                "--secret_tenant_sp ${{inputs.secret_tenant_sp}} "
                                "--secret_client_sp ${{inputs.secret_client_sp}} "
                                "--secret_sp ${{inputs.secret_sp}} "
                                "--pk_secret ${{inputs.pk_secret}} "
                                "--pk_pass_secret ${{inputs.pk_pass_secret}} "
                                "--pubk_secret ${{inputs.pubk_secret}} "
                                "$[[--whisper_model_name ${{inputs.whisper_model_name}}]] "
                                "$[[--num_workers ${{inputs.num_workers}}]] "
                                "$[[--beam_size ${{inputs.beam_size}}]] "
                                "$[[--word_level_timestamps ${{inputs.word_level_timestamps}}]] "
                                "$[[--condition_on_previous_text ${{inputs.condition_on_previous_text}}]] "
                                "$[[--compute_type ${{inputs.compute_type}}]] "
                                "$[[--language_code ${{inputs.language_code}}]] "
                                "--output_asr_path ${{outputs.output_asr_path}} "
                                f"--allowed_failed_percent {config_dct['job']['allowed_failed_percent']} "
                                f"--progress_update_timeout {config_dct['job']['progress_update_timeout']} "
                                f"--task_overhead_timeout {config_dct['job']['task_overhead_timeout']} "
                                f"--first_task_creation_timeout {config_dct['job']['first_task_creation_timeout']} "
                                f"--resource_monitor_interval {config_dct['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )

    #
    # Declare Parallel task to perform forced alignment, based on Viterbi algorithm by nvidia implementation
    # For detailed info, check: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=python
    #
    nfa_component = parallel_run_function(
        name="pNFA",
        display_name="Parallel forced alignment",
        description="Parallel component to perform NFA on a large amount of audios",
        inputs=dict(
            input_audio_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be transcribed"),
            input_asr_path=Input(type=AssetTypes.URI_FOLDER, description="Transcriptions of audios to be analysed"),
            keyvault_name=Input(type="string"),
            secret_tenant_sp=Input(type="string"),
            secret_client_sp=Input(type="string"),
            secret_sp=Input(type="string"),
            pk_secret=Input(type="string"),
            pk_pass_secret=Input(type="string"),
            pubk_secret=Input(type="string"),
            nfa_model_name=Input(type="string", default=config_dct['fa']['model_name'], optional=True),
            batch_size=Input(type="integer", default=config_dct['fa']['batch_size'], optional=True)
        ),
        outputs=dict(output_fa_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=config_dct['job']['instance_count'],
        max_concurrency_per_instance=config_dct['job']['max_concurrency_per_instance'],
        mini_batch_size=config_dct['job']['mini_batch_size'],
        mini_batch_error_threshold=config_dct['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=config_dct['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=config_dct['job']['max_retries'],
            timeout=config_dct['job']['timeout']
        ), 
        task=RunFunction(
            code="../components/nfa/src",
            entry_script="main.py",
            environment=ml_client.environments.get(name="nfa_env", version="1"),
            program_arguments="--input_audio_path ${{inputs.input_audio_path}} "
                                "--input_asr_path ${{inputs.input_asr_path}} "
                                "--keyvault_name ${{inputs.keyvault_name}} "
                                "--secret_tenant_sp ${{inputs.secret_tenant_sp}} "
                                "--secret_client_sp ${{inputs.secret_client_sp}} "
                                "--secret_sp ${{inputs.secret_sp}} "
                                "--pk_secret ${{inputs.pk_secret}} "
                                "--pk_pass_secret ${{inputs.pk_pass_secret}} "
                                "--pubk_secret ${{inputs.pubk_secret}} "
                                "$[[--nfa_model_name ${{inputs.nfa_model_name}}]] "
                                "$[[--batch_size ${{inputs.batch_size}}]] "
                                "--output_fa_path ${{outputs.output_fa_path}} "
                                f"--allowed_failed_percent {config_dct['job']['allowed_failed_percent']} "
                                f"--progress_update_timeout {config_dct['job']['progress_update_timeout']} "
                                f"--task_overhead_timeout {config_dct['job']['task_overhead_timeout']} "
                                f"--first_task_creation_timeout {config_dct['job']['first_task_creation_timeout']} "
                                f"--resource_monitor_interval {config_dct['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )

    #
    # Declare Parallel task to perform speaker diarization
    # For detailed info, check: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=python
    #
    diar_component = parallel_run_function(
        name="pMSDD",
        display_name="Parallel diarization",
        description="Parallel component to perform speaker diarization on a large amount of audios",
        inputs=dict(
            input_audio_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be diarized"),
            input_asr_path=Input(type=AssetTypes.URI_FOLDER, description="Transcriptions of those audios"),
            keyvault_name=Input(type="string"),
            secret_tenant_sp=Input(type="string"),
            secret_client_sp=Input(type="string"),
            secret_sp=Input(type="string"),
            pk_secret=Input(type="string"),
            pk_pass_secret=Input(type="string"),
            pubk_secret=Input(type="string"),
            event_type=Input(type="string", default=config_dct['diarization']['event_type'], optional=True),
            max_num_speakers=Input(type="integer", default=config_dct['diarization']['max_num_speakers'], optional=True),
            min_window_length=Input(type="number", default=config_dct['diarization']['min_window_length'], optional=True),
            overlap_threshold=Input(type="number", default=config_dct['diarization']['overlap_threshold'], optional=True)
        ),
        outputs=dict(output_diar_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=config_dct['job']['instance_count'],
        max_concurrency_per_instance=config_dct['job']['max_concurrency_per_instance'],
        mini_batch_size=config_dct['job']['mini_batch_size'],
        mini_batch_error_threshold=config_dct['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=config_dct['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=config_dct['job']['max_retries'],
            timeout=config_dct['job']['timeout']
        ), 
        task=RunFunction(
            code="../components/diar/src",
            entry_script="main.py",
            environment=ml_client.environments.get(name="diar_env", version="1"),
            program_arguments="--input_audio_path ${{inputs.input_audio_path}} "
                                "--input_asr_path ${{inputs.input_asr_path}} "
                                "--keyvault_name ${{inputs.keyvault_name}} "
                                "--secret_tenant_sp ${{inputs.secret_tenant_sp}} "
                                "--secret_client_sp ${{inputs.secret_client_sp}} "
                                "--secret_sp ${{inputs.secret_sp}} "
                                "--pk_secret ${{inputs.pk_secret}} "
                                "--pk_pass_secret ${{inputs.pk_pass_secret}} "
                                "--pubk_secret ${{inputs.pubk_secret}} "
                                "$[[--event_type ${{inputs.event_type}}]] "
                                "$[[--max_num_speakers ${{inputs.max_num_speakers}}]] "
                                "$[[--min_window_length ${{inputs.min_window_length}}]] "
                                "$[[--overlap_threshold ${{inputs.overlap_threshold}}]] "
                                "--output_diar_path ${{outputs.output_diar_path}} "
                                f"--allowed_failed_percent {config_dct['job']['allowed_failed_percent']} "
                                f"--progress_update_timeout {config_dct['job']['progress_update_timeout']} "
                                f"--task_overhead_timeout {config_dct['job']['task_overhead_timeout']} "
                                f"--first_task_creation_timeout {config_dct['job']['first_task_creation_timeout']} "
                                f"--resource_monitor_interval {config_dct['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )

    #
    # Declare Parallel task to perform merge and alignment
    # For detailed info, check: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=python
    #
    ma_component = parallel_run_function(
        name="pMA",
        display_name="Parallel merge & alignment",
        description="Parallel component to align transcriptions and diarization",
        inputs=dict(
            input_asr_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be diarized"),
            input_diar_path=Input(type=AssetTypes.URI_FOLDER, description="Transcriptions of those audios"),
            keyvault_name=Input(type="string"),
            secret_tenant_sp=Input(type="string"),
            secret_client_sp=Input(type="string"),
            secret_sp=Input(type="string"),
            pk_secret=Input(type="string"),
            pk_pass_secret=Input(type="string"),
            pubk_secret=Input(type="string"),
            cosmosdb_name=Input(type="string"),
            cosmosdb_collection=Input(type="string"),
            cosmosdb_cs_secret=Input(type="string"),
            ner_chunk_size=Input(type="integer", default=config_dct['align']['ner_chunk_size'], optional=True),
            ner_stride=Input(type="integer", default=config_dct['align']['ner_stride'], optional=True),
            max_words_in_sentence=Input(type="integer", default=config_dct['align']['max_words_in_sentence'], optional=True)
        ),
        outputs=dict(output_sm_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=config_dct['job']['instance_count'],
        max_concurrency_per_instance=config_dct['job']['max_concurrency_per_instance'],
        mini_batch_size=config_dct['job']['mini_batch_size'],
        mini_batch_error_threshold=config_dct['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=config_dct['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=config_dct['job']['max_retries'],
            timeout=config_dct['job']['timeout']
        ), 
        task=RunFunction(
            code="../components/merge_align/src",
            entry_script="main.py",
            environment=ml_client.environments.get(name="merge_align_env", version="1"),          
            program_arguments="--input_asr_path ${{inputs.input_asr_path}} "
                                "--input_diar_path ${{inputs.input_diar_path}} "
                                "--keyvault_name ${{inputs.keyvault_name}} "
                                "--secret_tenant_sp ${{inputs.secret_tenant_sp}} "
                                "--secret_client_sp ${{inputs.secret_client_sp}} "
                                "--secret_sp ${{inputs.secret_sp}} "
                                "--pk_secret ${{inputs.pk_secret}} "
                                "--pk_pass_secret ${{inputs.pk_pass_secret}} "
                                "--pubk_secret ${{inputs.pubk_secret}} "
                                "--cosmosdb_name ${{inputs.cosmosdb_name}} "
                                "--cosmosdb_collection ${{inputs.cosmosdb_collection}} "
                                "--cosmosdb_cs_secret ${{inputs.cosmosdb_cs_secret}} "
                                "$[[--ner_chunk_size ${{inputs.ner_chunk_size}}]] "
                                "$[[--ner_stride ${{inputs.ner_stride}}]] "
                                "$[[--max_words_in_sentence ${{inputs.max_words_in_sentence}}]] "
                                "--output_sm_path ${{outputs.output_sm_path}} "
                                f"--allowed_failed_percent {config_dct['job']['allowed_failed_percent']} "
                                f"--progress_update_timeout {config_dct['job']['progress_update_timeout']} "
                                f"--task_overhead_timeout {config_dct['job']['task_overhead_timeout']} "
                                f"--first_task_creation_timeout {config_dct['job']['first_task_creation_timeout']} "
                                f"--resource_monitor_interval {config_dct['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )


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
        prep_node = prep_component(
            input_path=input_dts,
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
            demucs_model=demucs_model
        )
        prep_node.outputs.output_prep_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )
        prep_node.compute = aml_t4_cluster

        # ASR
        asr_node = asr_component(
            input_path = prep_node.outputs.output_prep_path,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            whisper_model_name = asr_model_name,
            num_workers = asr_num_workers,
            beam_size = asr_beam_size,
            word_level_timestamps = word_level_timestamps,
            condition_on_previous_text = condition_on_previous_text,
            compute_type = asr_compute_type,
            language_code = asr_language_code
        )
        asr_node.outputs.output_asr_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )
        asr_node.compute = aml_t4_cluster

        # NFA
        nfa_node = nfa_component(
            input_audio_path=prep_node.outputs.output_prep_path,
            input_asr_path=asr_node.outputs.output_asr_path,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            nfa_model_name=nfa_model_name,
            batch_size=nfa_batch_size
        )
        nfa_node.outputs.output_fa_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )
        nfa_node.compute = aml_t4_cluster

        # Diarization
        diar_node = diar_component(
            input_audio_path = prep_node.outputs.output_prep_path,
            input_asr_path = nfa_node.outputs.output_fa_path,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            event_type = diar_event_type,
            max_num_speakers = diar_max_num_speakers,
            min_window_length = diar_min_window_length,
            overlap_threshold = diar_overlap_threshold
        )
        diar_node.outputs.output_diar_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )
        diar_node.compute = aml_a100_cluster

        # Merge&Align
        ma_node = ma_component(
            input_asr_path = nfa_node.outputs.output_fa_path,
            input_diar_path = diar_node.outputs.output_diar_path,
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
            ner_chunk_size = ma_ner_chunk_size,
            ner_stride = ma_ner_stride,
            max_words_in_sentence = ma_max_words_in_sentence
        )
        ma_node.compute = aml_t4_cluster
        ma_node.outputs.output_sm_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )

        # Remove STT data
        rsttd_node = rsttd_comp(
            input_path = ma_node.outputs.output_sm_path,
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
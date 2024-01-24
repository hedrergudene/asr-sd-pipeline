# Libraries
import logging as log
import sys
import json
import re
from uuid import uuid4
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import RetrySettings
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.parallel import parallel_run_function, RunFunction

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global ml_client
    # Get credential token
    log.info("Get credential token:")
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")

    # Get a handle to workspace
    log.info("Set up ML Client:")
    ml_client = MLClient(
        credential=credential,
        subscription_id='61888e35-639c-452a-8e15-b59ca726dfb2',
        resource_group_name='RSGREU2IACGD01',
        workspace_name='azmleu2iacgd01'
    )


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input, run an AzureML pipeline and return a dummy metadata.
    raw_data = {
        "azure":{
            "cpu_cluster": "XXXXXX",
            "gpu_cluster_t4": "XXXXXX",
            "gpu_cluster_a100": "XXXXXX",
            "project_name": "XXXXXX",
            "storage_id": "XXXXXX",
            "container_name": "XXXXXX",
            "input_path": "XXXXXX",
            "output_path": "XXXXXX"
        },
        "keyvault": {
            "name": "XXXXXX",
            "secret_tenant_sp": "XXXXXX",
            "secret_client_sp": "XXXXXX",
            "secret_sp": "XXXXXX",
            "pk_secret": "XXXXXX",
            "pk_pass_secret": "XXXXXX",
            "pubk_secret": "XXXXXX"
        },
        "cosmosdb": {
            "name": "XXXXXX",
            "collection": "XXXXXX",
            "cs_secret": "XXXXXX",
        }
        "preprocessing": {
            "vad_threshold": 0.75,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "demucs_model": "htdemucs"
        },
        "asr": {
            "model_name": "large-v3",
            "num_workers": 4,
            "beam_size": 5,
            "word_level_timestamps": True,
            "condition_on_previous_text": True,
            "language_code": "es",
            "compute_type": "float16"
        },
        "fa": {
            "model_name": "stt_es_fastconformer_hybrid_large_pc",
            "batch_size": 16
        },
        "diarization": {
            "event_type": "telephonic",
            "max_num_speakers": 3,
            "min_window_length": 0.2,
            "overlap_threshold": 0.8
        },
        "align": {
            "ner_chunk_size": 80,
            "ner_stride": 5,
            "max_words_in_sentence": 60
        },
        "job": {
            "instance_count": 2,
            "max_concurrency_per_instance": 1,
            "mini_batch_size": "1",
            "mini_batch_error_threshold": 1,
            "error_threshold": 1,
            "max_retries": 2,
            "timeout": 3600,
            "allowed_failed_percent": 0,
            "progress_update_timeout": 60000,
            "task_overhead_timeout": 3600,
            "first_task_creation_timeout": 60000,
            "resource_monitor_interval": 0
        }
    }
    """
    # Read message
    raw_data = json.loads(raw_data)

    # Validations
    ## Azure (all values are mandatory)
    if raw_data.get('azure') is None:
        raise ValueError(
                f"Please send a request with keyvault key and, at least, 'input_path', 'output_path', " + \
                f"'storage_id', 'container_name', 'project_name', 'cpu_cluster', 'gpu_cluster_t4' and 'gpu_cluster_a100' values."
        )
    else:
        if ((raw_data['azure'].get('input_path') is None) | 
            (raw_data['azure'].get('output_path') is None) | 
            (raw_data['azure'].get('storage_id') is None) | 
            (raw_data['azure'].get('container_name') is None) |
            (raw_data['azure'].get('project_name') is None) |
            (raw_data['azure'].get('cpu_cluster') is None) |
            (raw_data['azure'].get('gpu_cluster_t4') is None) |
            (raw_data['azure'].get('gpu_cluster_a100') is None)
        ):
            raise ValueError(
                f"Please send a request with keyvault key and, at least, 'input_path', 'output_path', " + \
                f"'storage_id', 'container_name', 'project_name', 'cpu_cluster', 'gpu_cluster_t4' and 'gpu_cluster_a100' values."
            )
    raw_data['azure']['filepath'] = re.findall('azureml://datastores/mysh/paths/(.*)',raw_data['azure']['output_path'])[0]
    ## Keyvault (all values are mandatory)
    if raw_data.get('keyvault') is None:
        raise ValueError(
            f"Please send a request with keyvault key and, at least, 'name', 'secret_tenant_sp', " + \
            f"'secret_client_sp', 'secret_sp', 'pk_secret', 'pk_pass_secret' and 'pubk_secret' values."
        )
    else:
        if ((raw_data['keyvault'].get('name') is None) | 
            (raw_data['keyvault'].get('secret_tenant_sp') is None) | 
            (raw_data['keyvault'].get('secret_client_sp') is None) | 
            (raw_data['keyvault'].get('secret_sp') is None) |
            (raw_data['keyvault'].get('pk_secret') is None) |
            (raw_data['keyvault'].get('pk_pass_secret') is None) |
            (raw_data['keyvault'].get('pubk_secret') is None)
        ):
            raise ValueError(
                f"Please send a request with keyvault key and, at least, 'name', 'secret_tenant_sp', " + \
                f"'secret_client_sp', 'secret_sp', 'pk_secret', 'pk_pass_secret' and 'pubk_secret' values."
            )
    ## CosmosDB (all values are mandatory)
    if raw_data.get('cosmosdb') is None:
        raise ValueError(
            f"Please send a request with cosmosdb key and values 'name', 'collection' and 'cs_secret'."
        )
    else:
        if ((raw_data['keyvault'].get('name') is None) | 
            (raw_data['keyvault'].get('collection') is None) | 
            (raw_data['keyvault'].get('cs_secret') is None)
        ):
            raise ValueError(
                f"Please send a request with cosmosdb key and values 'name', 'collection' and 'cs_secret'."
            )
    ## Preprocessing (values are optional)
    if raw_data.get('preprocessing') is None:
        raw_data['preprocessing'] = {
            "vad_threshold": 0.75,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "demucs_model": "htdemucs"
        }
    else:
        if (raw_data['preprocessing'].get('vad_threshold') is None): raw_data['preprocessing']['vad_threshold'] = 0.75
        if (raw_data['preprocessing'].get('min_speech_duration_ms') is None): raw_data['preprocessing']['min_speech_duration_ms'] = 250
        if (raw_data['preprocessing'].get('min_silence_duration_ms') is None): raw_data['preprocessing']['min_silence_duration_ms'] = 500
        if (raw_data['preprocessing'].get('demucs_model') is None): raw_data['preprocessing']['demucs_model'] = "htdemucs"
    ## ASR (values are optional)
    if raw_data.get('asr') is None:
        raw_data['asr'] = {
            "model_name": "large-v3",
            "num_workers": 4,
            "beam_size": 5,
            "word_level_timestamps": True,
            "condition_on_previous_text": True,
            "language_code": "es",
            "compute_type": "float16"
        }
    else:
        if (raw_data['asr'].get('model_name') is None): raw_data['asr']['model_name'] = "large-v3"
        if (raw_data['asr'].get('num_workers') is None): raw_data['asr']['num_workers'] = 4
        if (raw_data['asr'].get('beam_size') is None): raw_data['asr']['beam_size'] = 5
        if (raw_data['asr'].get('word_level_timestamps') is None): raw_data['asr']['word_level_timestamps'] = True
        if (raw_data['asr'].get('condition_on_previous_text') is None): raw_data['asr']['condition_on_previous_text'] = True
        if (raw_data['asr'].get('language_code') is None): raw_data['asr']['language_code'] = "es"
        if (raw_data['asr'].get('compute_type') is None): raw_data['asr']['compute_type'] = "float16"
    ## NFA (values are optional)
    if raw_data.get('fa') is None:
        raw_data['fa'] = {
            "model_name": "stt_es_fastconformer_hybrid_large_pc",
            "batch_size": 16
        }
    else:
        if (raw_data['fa'].get('model_name') is None): raw_data['fa']['model_name'] = "stt_es_fastconformer_hybrid_large_pc"
        if (raw_data['fa'].get('batch_size') is None): raw_data['fa']['batch_size'] = 16
    ## Speaker diarization (values are optional)
    if raw_data.get('diar') is None:
        raw_data['diar'] = {
            "event_type": "telephonic",
            "max_num_speakers": 3,
            "min_window_length": 0.2,
            "overlap_threshold": 0.8
        }
    else:
        if (raw_data['diar'].get('event_type') is None): raw_data['diar']['event_type'] = "telephonic"
        if (raw_data['diar'].get('max_num_speakers') is None): raw_data['diar']['max_num_speakers'] = 3
        if (raw_data['diar'].get('min_window_length') is None): raw_data['diar']['min_window_length'] = 0.2
        if (raw_data['diar'].get('overlap_threshold') is None): raw_data['diar']['overlap_threshold'] = 0.8
    ## Sentence mapping (values are optional)
    if raw_data.get('align') is None:
        raw_data['align'] = {
            "ner_chunk_size": 40,
            "ner_stride": 5,
            "max_words_in_sentence": 60
        }
    else:
        if (raw_data['align'].get('ner_chunk_size') is None): raw_data['align']['ner_chunk_size'] = 40
        if (raw_data['align'].get('ner_stride') is None): raw_data['align']['ner_stride'] = 5
        if (raw_data['align'].get('max_words_in_sentence') is None): raw_data['align']['max_words_in_sentence'] = 60
    ## Job configuration (values are optional)
    if raw_data.get('job') is None:
        raw_data['job'] = {
            "instance_count": 2,
            "max_concurrency_per_instance": 1,
            "mini_batch_size": "16",
            "mini_batch_error_threshold": 1,
            "error_threshold": 1,
            "max_retries": 2,
            "timeout": 3600,
            "allowed_failed_percent": 0,
            "progress_update_timeout": 60000,
            "task_overhead_timeout": 3600,
            "first_task_creation_timeout": 60000,
            "resource_monitor_interval": 0
        }
    else:
        if (raw_data['job'].get('instance_count') is None): raw_data['job']['instance_count'] = 2
        if (raw_data['job'].get('max_concurrency_per_instance') is None): raw_data['job']['max_concurrency_per_instance'] = 1
        if (raw_data['job'].get('mini_batch_size') is None): raw_data['job']['mini_batch_size'] = "16"
        if (raw_data['job'].get('mini_batch_error_threshold') is None): raw_data['job']['mini_batch_error_threshold'] = 1
        if (raw_data['job'].get('error_threshold') is None): raw_data['job']['error_threshold'] = 1
        if (raw_data['job'].get('max_retries') is None): raw_data['job']['max_retries'] = 2
        if (raw_data['job'].get('timeout') is None): raw_data['job']['timeout'] = 3600
        if (raw_data['job'].get('allowed_failed_percent') is None): raw_data['job']['allowed_failed_percent'] = 0
        if (raw_data['job'].get('progress_update_timeout') is None): raw_data['job']['progress_update_timeout'] = 60000
        if (raw_data['job'].get('task_overhead_timeout') is None): raw_data['job']['task_overhead_timeout'] = 3600
        if (raw_data['job'].get('first_task_creation_timeout') is None): raw_data['job']['first_task_creation_timeout'] = 60000
        if (raw_data['job'].get('resource_monitor_interval') is None): raw_data['job']['resource_monitor_interval'] = 0


    # Fetch components
    rsttd_comp = load_component(source="./components/remove_stt_data/remove_stt_data.yaml")

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
            vad_threshold=Input(type="number"),
            min_speech_duration_ms=Input(type="integer"),
            min_silence_duration_ms=Input(type="integer"),
            demucs_model=Input(type="string")
        ),
        outputs=dict(
            output_prep_path=Output(type=AssetTypes.URI_FOLDER)
        ),
        input_data="${{inputs.input_path}}",
        instance_count=raw_data['job']['instance_count'],
        max_concurrency_per_instance=raw_data['job']['max_concurrency_per_instance'],
        mini_batch_size=raw_data['job']['mini_batch_size'],
        mini_batch_error_threshold=raw_data['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=raw_data['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=raw_data['job']['max_retries'],
            timeout=raw_data['job']['timeout']
        ), 
        task=RunFunction(
            code="./components/prep/src",
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
                              "--vad_threshold ${{inputs.vad_threshold}} "
                              "--min_speech_duration_ms ${{inputs.min_speech_duration_ms}} "
                              "--min_silence_duration_ms ${{inputs.min_silence_duration_ms}} "
                              "--demucs_model ${{inputs.demucs_model}} "
                              "--output_prep_path ${{outputs.output_prep_path}} "
                              f"--allowed_failed_percent {raw_data['job']['allowed_failed_percent']} "
                              f"--progress_update_timeout {raw_data['job']['progress_update_timeout']} "
                              f"--task_overhead_timeout {raw_data['job']['task_overhead_timeout']} "
                              f"--first_task_creation_timeout {raw_data['job']['first_task_creation_timeout']} "
                              f"--resource_monitor_interval {raw_data['job']['resource_monitor_interval']} ",
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
            whisper_model_name=Input(type="string"),
            num_workers=Input(type="integer"),
            beam_size=Input(type="integer"),
            word_level_timestamps=Input(type="boolean"),
            condition_on_previous_text=Input(type="boolean"),
            compute_type=Input(type="string"),
            language_code=Input(type="string")
        ),
        outputs=dict(output_asr_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_path}}",
        instance_count=raw_data['job']['instance_count'],
        max_concurrency_per_instance=raw_data['job']['max_concurrency_per_instance'],
        mini_batch_size=raw_data['job']['mini_batch_size'],
        mini_batch_error_threshold=raw_data['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=raw_data['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=raw_data['job']['max_retries'],
            timeout=raw_data['job']['timeout']
        ), 
        task=RunFunction(
            code="./components/asr/src",
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
                              "--whisper_model_name ${{inputs.whisper_model_name}} "
                              "--num_workers ${{inputs.num_workers}} "
                              "--beam_size ${{inputs.beam_size}} "
                              "--word_level_timestamps ${{inputs.word_level_timestamps}} "
                              "--condition_on_previous_text ${{inputs.condition_on_previous_text}} "
                              "--compute_type ${{inputs.compute_type}} "
                              "--language_code ${{inputs.language_code}} "
                              "--output_asr_path ${{outputs.output_asr_path}} "
                              f"--allowed_failed_percent {raw_data['job']['allowed_failed_percent']} "
                              f"--progress_update_timeout {raw_data['job']['progress_update_timeout']} "
                              f"--task_overhead_timeout {raw_data['job']['task_overhead_timeout']} "
                              f"--first_task_creation_timeout {raw_data['job']['first_task_creation_timeout']} "
                              f"--resource_monitor_interval {raw_data['job']['resource_monitor_interval']} ",
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
            nfa_model_name=Input(type="string"),
            batch_size=Input(type="integer")
        ),
        outputs=dict(output_fa_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=raw_data['job']['instance_count'],
        max_concurrency_per_instance=raw_data['job']['max_concurrency_per_instance'],
        mini_batch_size=raw_data['job']['mini_batch_size'],
        mini_batch_error_threshold=raw_data['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=raw_data['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=raw_data['job']['max_retries'],
            timeout=raw_data['job']['timeout']
        ), 
        task=RunFunction(
            code="./components/nfa/src",
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
                              "--nfa_model_name ${{inputs.nfa_model_name}} "
                              "--batch_size ${{inputs.batch_size}} "
                              "--output_fa_path ${{outputs.output_fa_path}} "
                              f"--allowed_failed_percent {raw_data['job']['allowed_failed_percent']} "
                              f"--progress_update_timeout {raw_data['job']['progress_update_timeout']} "
                              f"--task_overhead_timeout {raw_data['job']['task_overhead_timeout']} "
                              f"--first_task_creation_timeout {raw_data['job']['first_task_creation_timeout']} "
                              f"--resource_monitor_interval {raw_data['job']['resource_monitor_interval']} ",
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
            event_type=Input(type="string"),
            max_num_speakers=Input(type="integer"),
            min_window_length=Input(type="number"),
            overlap_threshold=Input(type="number")
        ),
        outputs=dict(output_diar_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=raw_data['job']['instance_count'],
        max_concurrency_per_instance=raw_data['job']['max_concurrency_per_instance'],
        mini_batch_size=raw_data['job']['mini_batch_size'],
        mini_batch_error_threshold=raw_data['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=raw_data['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=raw_data['job']['max_retries'],
            timeout=raw_data['job']['timeout']
        ), 
        task=RunFunction(
            code="./components/diar/src",
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
                              "--event_type ${{inputs.event_type}} "
                              "--max_num_speakers ${{inputs.max_num_speakers}} "
                              "--min_window_length ${{inputs.min_window_length}} "
                              "--overlap_threshold ${{inputs.overlap_threshold}} "
                              "--output_diar_path ${{outputs.output_diar_path}} "
                              f"--allowed_failed_percent {raw_data['job']['allowed_failed_percent']} "
                              f"--progress_update_timeout {raw_data['job']['progress_update_timeout']} "
                              f"--task_overhead_timeout {raw_data['job']['task_overhead_timeout']} "
                              f"--first_task_creation_timeout {raw_data['job']['first_task_creation_timeout']} "
                              f"--resource_monitor_interval {raw_data['job']['resource_monitor_interval']} ",
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
            ner_chunk_size=Input(type="integer"),
            ner_stride=Input(type="integer"),
            max_words_in_sentence=Input(type="integer")
        ),
        outputs=dict(output_sm_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_asr_path}}",
        instance_count=raw_data['job']['instance_count'],
        max_concurrency_per_instance=raw_data['job']['max_concurrency_per_instance'],
        mini_batch_size=raw_data['job']['mini_batch_size'],
        mini_batch_error_threshold=raw_data['job']['mini_batch_error_threshold'],
        logging_level="DEBUG",
        error_threshold=raw_data['job']['error_threshold'],
        retry_settings=RetrySettings(
            max_retries=raw_data['job']['max_retries'],
            timeout=raw_data['job']['timeout']
        ), 
        task=RunFunction(
            code="./components/merge_align/src",
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
                              "--ner_chunk_size ${{inputs.ner_chunk_size}} "
                              "--ner_stride ${{inputs.ner_stride}} "
                              "--max_words_in_sentence ${{inputs.max_words_in_sentence}} "
                              "--output_sm_path ${{outputs.output_sm_path}} "
                              f"--allowed_failed_percent {raw_data['job']['allowed_failed_percent']} "
                              f"--progress_update_timeout {raw_data['job']['progress_update_timeout']} "
                              f"--task_overhead_timeout {raw_data['job']['task_overhead_timeout']} "
                              f"--first_task_creation_timeout {raw_data['job']['first_task_creation_timeout']} "
                              f"--resource_monitor_interval {raw_data['job']['resource_monitor_interval']} ",
            # All values output by run() method invocations will be aggregated into one unique file which is created in the output location.
            # If it is not set, 'summary_only' would invoked, which means user script is expected to store the output itself.
            #append_row_to="${{outputs.output_path}}"
        ),
    )


    #
    # Create pipeline
    #

    # Set the input and output URI paths for the data.
    input_dts = Input(
        path=raw_data['azure']['input_path'],
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.RO_MOUNT #Alternative, DOWNLOAD
    )

    output_dts = Output(
        path=raw_data['azure']['output_path'],
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.RW_MOUNT
    )

    @pipeline(
        default_compute=raw_data['azure']['gpu_cluster_t4']
    )
    def parallel_job():
        
        # Preprocessing
        prep_node = prep_component(
            input_path=input_dts,
            keyvault_name=raw_data['keyvault']['name'],
            secret_tenant_sp=raw_data['keyvault']['secret_tenant_sp'],
            secret_client_sp=raw_data['keyvault']['secret_client_sp'],
            secret_sp=raw_data['keyvault']['secret_sp'],
            pk_secret=raw_data['keyvault']['pk_secret'],
            pk_pass_secret=raw_data['keyvault']['pk_pass_secret'],
            pubk_secret=raw_data['keyvault']['pubk_secret'],
            cosmosdb_name=raw_data['cosmosdb']['name'],
            cosmosdb_collection=raw_data['cosmosdb']['collection'],
            cosmosdb_cs_secret=raw_data['cosmosdb']['cs_secret'],
            vad_threshold=raw_data['preprocessing']['vad_threshold'],
            min_speech_duration_ms=raw_data['preprocessing']['min_speech_duration_ms'],
            min_silence_duration_ms=raw_data['preprocessing']['min_silence_duration_ms'],
            demucs_model=raw_data['preprocessing']['demucs_model']
        )
        prep_node.outputs.output_prep_path = output_dts

        # ASR
        asr_node = asr_component(
            input_path = prep_node.outputs.output_prep_path,
            keyvault_name=raw_data['keyvault']['name'],
            secret_tenant_sp=raw_data['keyvault']['secret_tenant_sp'],
            secret_client_sp=raw_data['keyvault']['secret_client_sp'],
            secret_sp=raw_data['keyvault']['secret_sp'],
            pk_secret=raw_data['keyvault']['pk_secret'],
            pk_pass_secret=raw_data['keyvault']['pk_pass_secret'],
            pubk_secret=raw_data['keyvault']['pubk_secret'],
            whisper_model_name = raw_data['asr']['model_name'],
            num_workers = raw_data['asr']['num_workers'],
            beam_size = raw_data['asr']['beam_size'],
            word_level_timestamps = raw_data['asr']['word_level_timestamps'],
            condition_on_previous_text = raw_data['asr']['condition_on_previous_text'],
            compute_type = raw_data['asr']['compute_type'],
            language_code = raw_data['asr']['language_code']
        )
        asr_node.outputs.output_asr_path = output_dts

        # NFA
        nfa_node = nfa_component(
            input_audio_path=prep_node.outputs.output_prep_path,
            input_asr_path=asr_node.outputs.output_asr_path,
            keyvault_name=raw_data['keyvault']['name'],
            secret_tenant_sp=raw_data['keyvault']['secret_tenant_sp'],
            secret_client_sp=raw_data['keyvault']['secret_client_sp'],
            secret_sp=raw_data['keyvault']['secret_sp'],
            pk_secret=raw_data['keyvault']['pk_secret'],
            pk_pass_secret=raw_data['keyvault']['pk_pass_secret'],
            pubk_secret=raw_data['keyvault']['pubk_secret'],
            nfa_model_name=raw_data['fa']['model_name'],
            batch_size=raw_data['fa']['batch_size']     
        )
        nfa_node.outputs.output_fa_path = output_dts

        # Diarization
        diar_node = diar_component(
            input_audio_path = prep_node.outputs.output_prep_path,
            input_asr_path = nfa_node.outputs.output_fa_path,
            keyvault_name=raw_data['keyvault']['name'],
            secret_tenant_sp=raw_data['keyvault']['secret_tenant_sp'],
            secret_client_sp=raw_data['keyvault']['secret_client_sp'],
            secret_sp=raw_data['keyvault']['secret_sp'],
            pk_secret=raw_data['keyvault']['pk_secret'],
            pk_pass_secret=raw_data['keyvault']['pk_pass_secret'],
            pubk_secret=raw_data['keyvault']['pubk_secret'],
            event_type = raw_data['diarization']['event_type'],
            max_num_speakers = raw_data['diarization']['max_num_speakers'],
            min_window_length = raw_data['diarization']['min_window_length'],
            overlap_threshold = raw_data['diarization']['overlap_threshold']
        )
        diar_node.outputs.output_diar_path = output_dts
        diar_node.compute = raw_data['azure']['gpu_cluster_a100']

        # Merge&Align
        ma_node = ma_component(
            input_asr_path = nfa_node.outputs.output_fa_path,
            input_diar_path = diar_node.outputs.output_diar_path,
            keyvault_name=raw_data['keyvault']['name'],
            secret_tenant_sp=raw_data['keyvault']['secret_tenant_sp'],
            secret_client_sp=raw_data['keyvault']['secret_client_sp'],
            secret_sp=raw_data['keyvault']['secret_sp'],
            pk_secret=raw_data['keyvault']['pk_secret'],
            pk_pass_secret=raw_data['keyvault']['pk_pass_secret'],
            pubk_secret=raw_data['keyvault']['pubk_secret'],
            cosmosdb_name=raw_data['cosmosdb']['name'],
            cosmosdb_collection=raw_data['cosmosdb']['collection'],
            cosmosdb_cs_secret=raw_data['cosmosdb']['cs_secret'],
            ner_chunk_size = raw_data['align']['ner_chunk_size'],
            ner_stride = raw_data['align']['ner_stride'],
            max_words_in_sentence = raw_data['align']['max_words_in_sentence']
        )
        ma_node.outputs.output_sm_path = output_dts
        diar_node.compute = raw_data['azure']['cpu_cluster']

        # Remove STT data
        rsttd_node = rsttd_comp(
            input_path = ma_node.outputs.output_sm_path,
            storage_id = raw_data['azure']['storage_id'],
            container_name = raw_data['azure']['container_name'],
            blob_filepath = raw_data['azure']['filepath']
        )
        rsttd_node.compute = raw_data['azure']['cpu_cluster']


    # Create a pipeline
    pipeline_job = parallel_job()
    # Run job
    seed_name = str(uuid4())
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=raw_data['azure']['project_name']+"_"+seed_name
    )

    return {'experiment_name': raw_data['azure']['project_name']+"_"+seed_name}
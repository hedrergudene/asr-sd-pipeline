# Libraries
import yaml
import sys
import os
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment, BuildContext, BatchRetrySettings, RetrySettings
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
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name'],
    )
        

    # Set the input and output URI paths for the data.
    input_audio_data = Input(
        path=config_dct['azure']['data_filepath'],
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.RO_MOUNT #Alternative, DOWNLOAD
    )

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
            vad_threshold=Input(type="number"),
            min_speech_duration_ms=Input(type="integer"),
            min_silence_duration_ms=Input(type="integer"),
            use_onnx_vad=Input(type="boolean"),
            demucs_model=Input(type="string")
        ),
        outputs=dict(
            output_audios_path=Output(type=AssetTypes.URI_FOLDER),
            output_metadata_path=Output(type=AssetTypes.URI_FOLDER)
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
            code="./components/prep/src",
            entry_script="main.py",
            environment=Environment(
                build=BuildContext(path=f"./components/prep/docker")
            ),
            program_arguments="--input_path ${{inputs.input_path}} "
                              "--vad_threshold ${{inputs.vad_threshold}} "
                              "--min_speech_duration_ms ${{inputs.min_speech_duration_ms}} "
                              "--min_silence_duration_ms ${{inputs.min_silence_duration_ms}} "
                              "--use_onnx_vad ${{inputs.use_onnx_vad}} "
                              "--demucs_model ${{inputs.demucs_model}} "
                              "--output_audios_path ${{outputs.output_audios_path}} "
                              "--output_metadata_path ${{outputs.output_metadata_path}} "
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
            input_audio_path=Input(type=AssetTypes.URI_FOLDER, description="Audios to be transcribed"),
            input_metadata_path=Input(type=AssetTypes.URI_FOLDER, description="Metadata attached to audios to be transcribed"),
            whisper_model_name=Input(type="string"),
            beam_size=Input(type="integer"),
            word_level_timestamps=Input(type="boolean"),
            condition_on_previous_text=Input(type="boolean"),
            compute_type=Input(type="string"),
            language_code=Input(type="string")
        ),
        outputs=dict(output_path=Output(type=AssetTypes.URI_FOLDER)),
        input_data="${{inputs.input_metadata_path}}",
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
            code="./components/asr/src",
            entry_script="main.py",
            environment=Environment(
                build=BuildContext(path=f"./components/asr/docker")
            ),
            program_arguments="--input_audio_path ${{inputs.input_audio_path}} "
                              "--input_metadata_path ${{inputs.input_metadata_path}} "
                              "--whisper_model_name ${{inputs.whisper_model_name}} "
                              "--beam_size ${{inputs.beam_size}} "
                              "--word_level_timestamps ${{inputs.word_level_timestamps}} "
                              "--condition_on_previous_text ${{inputs.condition_on_previous_text}} "
                              "--compute_type ${{inputs.compute_type}} "
                              "--language_code ${{inputs.language_code}} "
                              "--output_path ${{outputs.output_path}} "
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
            nfa_model_name=Input(type="string"),
            batch_size=Input(type="integer")
        ),
        outputs=dict(output_path=Output(type=AssetTypes.URI_FOLDER)),
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
            code="./components/nfa/src",
            entry_script="main.py",
            environment=Environment(
                build=BuildContext(path=f"./components/nfa/docker")
            ),
            program_arguments="--input_audio_path ${{inputs.input_audio_path}} "
                              "--input_asr_path ${{inputs.input_asr_path}} "
                              "--nfa_model_name ${{inputs.nfa_model_name}} "
                              "--batch_size ${{inputs.batch_size}} "
                              "--output_path ${{outputs.output_path}} "
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
            event_type=Input(type="string"),
            max_num_speakers=Input(type="integer")
        ),
        outputs=dict(output_path=Output(type=AssetTypes.URI_FOLDER)),
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
            code="./components/diar/src",
            entry_script="main.py",
            environment=Environment(
                build=BuildContext(path=f"./components/diar/docker")
            ),
            program_arguments="--input_audio_path ${{inputs.input_audio_path}} "
                              "--input_asr_path ${{inputs.input_asr_path}} "
                              "--event_type ${{inputs.event_type}} "
                              "--max_num_speakers ${{inputs.max_num_speakers}} "
                              "--output_path ${{outputs.output_path}} "
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
            max_words_in_sentence=Input(type="integer")
        ),
        outputs=dict(output_path=Output(type=AssetTypes.URI_FOLDER)),
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
            code="./components/merge_align/src",
            entry_script="main.py",
            environment=Environment(
                build=BuildContext(path=f"./components/merge_align/docker")
            ),          
            program_arguments="--input_asr_path ${{inputs.input_asr_path}} "
                              "--input_diar_path ${{inputs.input_diar_path}} "
                              "--max_words_in_sentence ${{inputs.max_words_in_sentence}} "
                              "--output_path ${{outputs.output_path}} "
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
    @pipeline(
        default_compute=config_dct['azure']['computing']['gpu_cluster_aml_id']
    )
    def parallel_job(input_dts):
        
        # Preprocessing
        prep_node = prep_component(
            input_path=input_dts,
            vad_threshold=config_dct['preprocessing']['vad_threshold'],
            min_speech_duration_ms=config_dct['preprocessing']['min_speech_duration_ms'],
            min_silence_duration_ms=config_dct['preprocessing']['min_silence_duration_ms'],
            use_onnx_vad=config_dct['preprocessing']['use_onnx_vad'],
            demucs_model=config_dct['preprocessing']['demucs_model']
        )

        # ASR
        asr_node = asr_component(
            input_audio_path = prep_node.outputs.output_audios_path,
            input_metadata_path = prep_node.outputs.output_metadata_path,
            whisper_model_name = config_dct['asr']['model_name'],
            beam_size = config_dct['asr']['beam_size'],
            word_level_timestamps = config_dct['asr']['word_level_timestamps'],
            condition_on_previous_text = config_dct['asr']['condition_on_previous_text'],
            compute_type = config_dct['asr']['compute_type'],
            language_code = config_dct['asr']['language_code']
        )

        # NFA
        nfa_node = nfa_component(
            input_audio_path=prep_node.outputs.output_audios_path,
            input_asr_path=asr_node.outputs.output_path,
            nfa_model_name=config_dct['fa']['model_name'],
            batch_size=config_dct['fa']['batch_size']     
        )

        # Diarization
        diar_node = diar_component(
            input_audio_path = prep_node.outputs.output_audios_path,
            input_asr_path = nfa_node.outputs.output_path,
            event_type = config_dct['diarization']['event_type'],
            max_num_speakers = config_dct['diarization']['max_num_speakers']
        )

        # Merge&Align
        ma_node = ma_component(
            input_asr_path = nfa_node.outputs.output_path,
            input_diar_path = diar_node.outputs.output_path,
            max_words_in_sentence = config_dct['align']['max_words_in_sentence']
        )

        return {
            "output": ma_node.outputs.output_path
        }

    # Create a pipeline
    pipeline_job = parallel_job(input_audio_data)
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=config_dct['azure']['experiment_name']
    )


if __name__=="__main__":
    fire.Fire(main)
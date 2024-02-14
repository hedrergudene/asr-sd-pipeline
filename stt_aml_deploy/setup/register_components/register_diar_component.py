# Libraries
import yaml
import sys
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output
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
    config_path='.../config/parallel_job.yaml'
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
            code=".../components/diar/src",
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
    def diar(
        input_dts:Input(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RO_MOUNT),
        input_asr:Input(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RO_MOUNT),
        output_dts:Input(type='string'),
        compute_cluster:Input(type="string"),
        keyvault_name:Input(type="string"),
        secret_tenant_sp:Input(type="string"),
        secret_client_sp:Input(type="string"),
        secret_sp:Input(type="string"),
        pk_secret:Input(type="string"),
        pk_pass_secret:Input(type="string"),
        pubk_secret:Input(type="string"),
        event_type:Input(type="string", default=config_dct['diarization']['event_type'], optional=True),
        max_num_speakers:Input(type="integer", default=config_dct['diarization']['max_num_speakers'], optional=True),
        min_window_length:Input(type="number", default=config_dct['diarization']['min_window_length'], optional=True),
        overlap_threshold:Input(type="number", default=config_dct['diarization']['overlap_threshold'], optional=True)
    ):

        diar_node = diar_component(
            input_audio_path = input_dts,
            input_asr_path = input_asr,
            keyvault_name=keyvault_name,
            secret_tenant_sp=secret_tenant_sp,
            secret_client_sp=secret_client_sp,
            secret_sp=secret_sp,
            pk_secret=pk_secret,
            pk_pass_secret=pk_pass_secret,
            pubk_secret=pubk_secret,
            event_type = event_type,
            max_num_speakers = max_num_speakers,
            min_window_length = min_window_length,
            overlap_threshold = overlap_threshold
        )
        diar_node.outputs.output_diar_path = Output(
            path=output_dts,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
            )
        diar_node.compute = compute_cluster

        return {'output_dts': diar_node.outputs.output_diar_path}


    # Create a pipeline
    pipeline_job = diar()

    # Component register
    ml_client.components.create_or_update(pipeline_job.component, version="1")


if __name__=="__main__":
    fire.Fire(main)
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
            code=".../components/prep/src",
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


    @pipeline(default_compute=config_dct['aml']['computing']['gpu_cluster_t4'])
    def prep(
        input_dts:Input(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RO_MOUNT),
        output_dts:Input(type="string"),
        aml_compute:Input(type="string"),
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
        demucs_model:Input(type="string", default=config_dct['preprocessing']['demucs_model'], optional=True)
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
        prep_node.compute = aml_compute

        return {'output_dts': prep_node.outputs.output_prep_path}

    # Create a pipeline
    pipeline_job = prep()

    # Component register
    ml_client.components.create_or_update(pipeline_job.component, version="1")


if __name__=="__main__":
    fire.Fire(main)
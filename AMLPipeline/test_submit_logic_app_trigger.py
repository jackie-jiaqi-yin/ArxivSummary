import os
# set the working directory as parent's of the script
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

from azure.ai.ml import MLClient, load_component, Input, Output
from azure.identity import  DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

# Import our Azure configuration helper for local execution
from AMLPipeline.azure_config_local import get_azure_ml_client, get_datastore_path

logic_app_trigger_component = load_component(source = 'AMLPipeline/logic_app_trigger/logic_app_trigger.yaml')
cpu_compute_target = os.getenv('AZURE_COMPUTE_CLUSTER_NAME', 'cpu-cluster')

dir = get_datastore_path("ArxivSummary")
input_dir = f'{dir}/current/'
output_dir = f'{dir}/current/'

@pipeline(
     default_compute=cpu_compute_target,
     description = 'azure logic app trigger test' 
)
def test_logic_app_trigger_pipeline(input_dir):
    logic_app_trigger_node = logic_app_trigger_component(
        input_dir=input_dir
    )
    logic_app_trigger_node.outputs.output_dir = Output(
        path=output_dir,
        type='uri_folder',
        mode='rw_mount'
    )

pipeline_job = test_logic_app_trigger_pipeline(
    input_dir=Input(path=input_dir, type='uri_folder', mode='ro_mount')
)


# Get ML client using environment variables
ml_client = get_azure_ml_client()

ml_client.jobs.create_or_update(pipeline_job, experiment_name='test_logic_app_trigger_pipeline',)                    
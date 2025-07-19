import os
# set the working directory as parent's of the script
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

from azure.ai.ml import MLClient, load_component, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

# Import our Azure configuration helper for local execution
from AMLPipeline.azure_config_local import get_azure_ml_client, get_datastore_path

copy_data_component = load_component(source = 'AMLPipeline/post_process/post_process.yaml')
cpu_compute_target = os.getenv('AZURE_COMPUTE_CLUSTER_NAME', 'cpu-cluster')
dir = get_datastore_path("ArxivSummary")
input_dir = f'{dir}/current/'
output_dir = f'{dir}/history/'

@pipeline(
    default_compute=cpu_compute_target,
    description='copy data from current folder to another folder'
)
def test_post_process_pipeline(input_dir):
    copy_data_node = copy_data_component(
        input_dir=input_dir
    )
    copy_data_node.outputs.output_dir = Output(
        path=output_dir,
        type='uri_folder',
        mode='rw_mount'
    )

pipeline_job = test_post_process_pipeline(
    input_dir=Input(path=input_dir, type='uri_folder', mode='ro_mount')
)

# Get ML client using environment variables
ml_client = get_azure_ml_client()

ml_client.jobs.create_or_update(pipeline_job, experiment_name='test_post_process_pipeline',)
import os
# set the working directory as parent's of the script
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

from azure.ai.ml import MLClient, load_component, Input, Output
from azure.identity import  DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

# Import our Azure configuration helper for local execution
from AMLPipeline.azure_config_local import get_azure_ml_client, get_datastore_path

arxiv_crawl_component = load_component(source = 'AMLPipeline/arxiv_crawl/arxiv_crawl.yaml')
cpu_compute_target = os.getenv('AZURE_COMPUTE_CLUSTER_NAME', 'cpu-cluster')
dir = get_datastore_path("ArxivSummary")
arxiv_save_dir = f'{dir}/current/'
latest_paper = 200
arxiv_query = """(cat:cs.CL OR cat:cs.AI) AND (ti:"large language model" OR abs:"large language model" OR ti:LLM OR abs:LLM)"""


@pipeline(
     default_compute=cpu_compute_target,
     description = 'llm paper summary' 
)
def test_arxiv_crawl_pipline(max_results, query):
    arxiv_crawl_node = arxiv_crawl_component(max_results = max_results, query = query)
    return {
        'output_dir': arxiv_crawl_node.outputs.output_dir
    }

pipeline_job = test_arxiv_crawl_pipline(max_results=latest_paper, query=arxiv_query)

# change the output path 
pipeline_job.outputs.output_dir = Output(
    path = arxiv_save_dir,
    type = 'uri_folder',
    mode = 'rw_mount'
)

# Get ML client using environment variables
ml_client = get_azure_ml_client()

ml_client.jobs.create_or_update(pipeline_job, experiment_name='test_data_loader_pipeline',)                    
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.ai.ml import MLClient, load_component, Input, Output
from azure.identity import  DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

# Import our Azure configuration helper for local execution
from AMLPipeline.azure_config_local import get_azure_ml_client, get_datastore_path

# load components
arxiv_crawl_component = load_component(source = 'AMLPipeline/arxiv_crawl/arxiv_crawl.yaml')
summary_component = load_component(source = 'AMLPipeline/summary/summary.yaml')
post_process_component = load_component(source = 'AMLPipeline/post_process/post_process.yaml')
logic_app_trigger_component = load_component(source = 'AMLPipeline/logic_app_trigger/logic_app_trigger.yaml')

cpu_compute_target = os.getenv('AZURE_COMPUTE_CLUSTER_NAME', 'cpu-cluster')

# Load datastore paths from environment variables
base_path = 'ArxivSummary_AI_Agent'
arxiv_save_dir = f'{get_datastore_path(base_path)}/current/'
summary_save_dir = f'{get_datastore_path(base_path)}/current/'
output_dir = f'{get_datastore_path(base_path)}/history/'

# Input parameters
## Airx crawl nodes
latest_paper = 50
arxiv_query =   """(cat:cs.CL OR cat:cs.AI or cat:cs.LG or cat:cs.HC or cat:cs.SE) AND (ti:"ai agent" OR abs:"ai agent" OR ti:"agentic ai" OR abs:"agentic ai")"""

## Summary nodes
system_query = """**You are an expert in natural language processing and language models. Your task is to analyze a collection of paper abstracts in this field, provided in a markdown file containing the paper title, authors, abstract, and PDF URL for each paper. Instead of summarizing individual abstracts, focus on synthesizing information across all papers to provide a comprehensive overview of the research landscape.**

    **General Instructions:**

    - **Avoid Redundancy:** Ensure that each section provides unique insights and does not repeat information from other sections.
    - **Diversity of Examples:** Use a diverse set of papers in each section to showcase as many different studies as possible.
    - **Limit Repetition of Papers:** Aim to mention each paper in only one section unless it is essential to illustrate a unique point.
    - **Focus on Section Objectives:** Adhere closely to the specific goals of each section, providing content that aligns with its unique focus.

    ---

    **Perform the following tasks and keep the structure of the markdown file as described below:**

    ---

    ## **Key Research Themes**

    - **Objective:** Identify and describe in detail 4-6 major research themes or focus areas that emerge from the abstracts.
    - **Instructions:**
      - Provide a comprehensive explanation of each theme, its significance in the field, and how it relates to broader goals in NLP and AI.
      - Discuss any subthemes or specific research questions within this area.
      - **Use Different Examples:** Mention 3-6 representative papers (include title and URL for each), ensuring these papers are not extensively discussed in other sections.
      - Explain how each paper contributes to or exemplifies the theme, without delving into methodological specifics covered in Section 5.

    ---
    ## **Methodological Approaches**

    - **Objective:** Identify 4-6 **common or emerging methodological approaches**  in the field that are prominent in the abstracts.
    - **Instructions:**
      - Provide a detailed explanation of each methodology, including its key components and underlying principles.
      - Discuss the advantages and potential limitations of this approach.
      - Explain how it differs from or improves upon previous methods.
      - **Highlight Different Papers:** List 3-4 papers that exemplify this approach (include titles and URLs), and briefly describe how each paper utilizes or advances the methodology. Avoid reusing papers from previous sections unless necessary.

    ---

    ## **Innovative or High-Impact Papers**

    - **Objective:** Identify 3-7 papers that appear to be the most innovative or potentially impactful, focusing on those not highlighted in previous sections.
    - **Instructions:**
      - For each paper, provide:
        - **a)** Paper title and URL.
        - **b)** A detailed explanation of its key innovation or potential impact, including:
          - The specific problem or challenge it addresses.
          - The novel approach or methodology it introduces.
          - The potential implications for the field or practical applications.
        - **c)** An analysis of how it relates to or advances one or more of the key themes, without repeating details from Section 2.
        - **d)** Any limitations or areas for future work mentioned in the abstract.

    ---

    ## **Challenges and Future Directions**

    - **Objective:** Identify 3-5 key challenges or open problems in the field, as evidenced by the abstracts, that have not been the primary focus of earlier sections.
    - **Instructions:**
      - Explain the nature of each problem and its importance to the field.
      - Discuss current approaches to addressing the challenge, citing relevant papers (with URLs).
      - Speculate on potential future directions for tackling these challenges.

    ---

    ## **Concluding Overview**

    - **Objective:** Provide a comprehensive (8-10 sentences) high-level summary of the current state and direction of research in language models and NLP.
    - **Instructions:**
      - Synthesize the key themes, trends, and innovations discussed earlier, without repeating specific details.
      - Offer insights into the overall trajectory of the field and potential future developments.
    ---

    **Final Reminders:**

    - **Synthesize Across Abstracts:** Focus on synthesizing information across all abstracts rather than summarizing individual papers.
    - **Deep Insights:** Provide deep insights into the collective body of research, highlighting connections between papers and themes.
    - **Unique Content:** Before finalizing, review each section to ensure that content is not duplicated elsewhere in the document.
    - **Expert Analysis:** Use your expertise to draw meaningful conclusions and provide context that goes beyond what's explicitly stated in the abstracts.
    - **Paper Citations:** Ensure that every mention of a specific paper includes its title and URL."""

batch_system_query = """**You are an expert in natural language processing and language models. Your task is to analyze a batch of paper abstracts and provide a comprehensive summary that will later be combined with other batch summaries.**. Whenever mention any paper, **include its title and URL**.

  **Focus on:**
  1. **Key Research Themes** - Identify 3-4 major themes in this batch
  2. **Methodological Approaches** - Highlight 2-3 main approaches used
  3. **Notable Papers** - Identify 2-3 most innovative or impactful papers
  4. **Technical Contributions** - Summarize main technical advances
  5. **Future Directions** - Any emerging trends or challenges mentioned

  **Keep the summary concise but comprehensive, as it will be combined with other batch summaries later.**"""
  
logic_app_name = os.getenv('LOGIC_APP_AGENTIC_AI_SECRET_NAME', 'agenticai-newsletter-logic-app-webhook-url')
blob_path_prefix = f"{os.getenv('AZURE_DATASTORE_NAME', 'workspaceblobstore')}/ArxivSummary_AI_Agent/current"

@pipeline(
    default_compute = cpu_compute_target,
    description = 'Agentit AI ArXiv paper summary'
)
def arxiv_summary_pipeline(max_results, 
                           query,
                           system_query,
                           batch_system_query,
                           logic_app_name,
                           blob_path_prefix):
    
    # crawl the paper from ArXiv
    arxiv_scrawl_node = arxiv_crawl_component(
        max_results = max_results,
        query = query
    )
    arxiv_scrawl_node.outputs.output_dir = Output(
        path = arxiv_save_dir,
        type = 'uri_folder',
        mode = 'rw_mount'
    )
    
    # summarize the papers
    summary_node = summary_component(
        input_dir = arxiv_scrawl_node.outputs.output_dir,
        system_query = system_query,
        batch_system_query = batch_system_query,
        model_name = 'gpt-4o',
        batch_size = 20,
        max_concurrent_batches = 5,
    )
    summary_node.outputs.output_dir = Output(
        path = summary_save_dir,
        type = 'uri_folder',
        mode = 'rw_mount'
    )

    # trigger logic app
    logic_app_trigger_node = logic_app_trigger_component(
        logic_app_name = logic_app_name,
        blob_path_prefix = blob_path_prefix,
        input_dir = summary_node.outputs.output_dir
    )
    logic_app_trigger_node.outputs.output_dir = Output(
        path = summary_save_dir,
        type = 'uri_folder',
        mode = 'rw_mount'
    )

    # copy data
    post_process_node = post_process_component(
        input_dir = summary_node.outputs.output_dir
    )
    post_process_node.outputs.output_dir = Output(
        path = output_dir,
        type = 'uri_folder',
        mode = 'rw_mount'
    )



# connect to the workspace
# Get ML client using environment variables
ml_client = get_azure_ml_client()

# create or update the pipline
pipeline_job = arxiv_summary_pipeline(
    max_results=latest_paper,
    query=arxiv_query,
    system_query=system_query,
    batch_system_query=batch_system_query,
    logic_app_name=logic_app_name,
    blob_path_prefix=blob_path_prefix
)

# submit the pipeline
submitted_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name='Agentic AI Arxiv paper Summary')
print(f"Job name: {submitted_job.name}")  
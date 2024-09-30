import os

from llama_index.core.llms import ChatMessage
from llama_index.llms.azure_openai import AzureOpenAI
from pathlib import Path
from src.utils import load_llm_config
import pandas as pd
from dotenv import load_dotenv


def _turn_metadata_to_md(metadata, output_dir):
    """
    Turn metadata to markdown file
    :param metadata: the dataframe containing the metadata
    :return: the str path to the markdown file
    """
    output_file = os.path.join(output_dir, 'metadata.md')
    with open(output_file, encoding='utf-8', mode='w') as f:
        for _, row in metadata.iterrows():
            title = row['title']
            authors = row['authors']
            abstract = row['abstract']
            url = row['pdf_url']
            f.write((f"## {title}\n\n"))
            f.write(f"**Authors**: {authors}\n\n")
            f.write(f"**Abstract**: {abstract}\n\n")
            f.write(f"**URL**: {url}\n\n")
    print(f"Metadata saved to {output_file}")
    return output_file


def summarize_abstracts(input_file, output_dir, summary_query):
    """
    Summarize all the abstracts with the summary_query
    :param input_file: the markdown file path containing the metadata
    :param output_dir: the directory to save the summaries
    :param summary_query: the query to summarize the abstracts
    :return: None
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    md_text = Path(input_file).read_text()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    load_dotenv()
    llm = AzureOpenAI(
        model_name='gpt-4o-0806',
        engine='gpt-4o',
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('GPT_API_VERSION'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )
    messages = [
        ChatMessage(role='system', content=summary_query),
        ChatMessage(role='user', content=f'Please summarize the abstracts: {md_text}')
    ]
    response = llm.chat(messages)
    # save
    with open(os.path.join(output_dir, 'abstract_summary.md'), 'w') as f:
        f.write(response.message.content)
    print(f"Summaries saved to {output_dir}/abstract_summary.md")


def summary_each_document(input_dir, summary_query):
    """
    Summarize documents in a directory
    :param input_dir: the directory containing the pdf

    :param summary_query: the query to summarize the documents
    :return: None
    """
    # connect to LLM and embedding API
    load_llm_config()
    # load documents
    pass


def summarize_papers(mode, input_file=None, input_dir=None, output_dir=None, system_query=None):
    """
    Summarize papers
    :param mode: the mode to summarize. It can be 'abstract' or 'document'
    :param input_file: the csv file containing the metadata, must exist if mode is 'abstract'
    :param input_dir: the directory containing the pdfs, must exist if mode is 'document'
    :param output_dir: the directory to save the summaries
    :param summary_query: the query to summarize the papers
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if mode == 'abstract':
        # turn the csv to markdown
        metadata = pd.read_csv(input_file)
        markdown_path = _turn_metadata_to_md(metadata, output_dir)
        summarize_abstracts(markdown_path, output_dir, system_query)
    elif mode == 'document':
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")
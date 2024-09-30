import os
from llama_index.core.llms import ChatMessage
from pathlib import Path
from src.utils import load_llm_config
import pandas as pd


def _turn_metadata_to_md(metadata, output_dir):
    """
    Turn metadata to markdown file
    :param metadata: the dataframe containing the metadata
    :return: the str path to the markdown file
    """
    output_file = os.path.join(output_dir, 'catalog.md')
    with open(output_file, encoding='utf-8', mode='w') as f:
        for _, row in metadata.iterrows():
            title = row['title']
            authors = row['authors']
            abstract = row['abstract']
            url = row['pdf_url']
            date = row['published'][0:10]
            f.write((f"## {title}\n\n"))
            f.write(f"**Authors**: {authors}\n\n")
            f.write(f"**Abstract**: {abstract}\n\n")
            f.write(f"**URL**: {url}\n\n")
            f.write(f"**Published**: {date}\n\n")
    print(f"Catalog saved to {output_file}")
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
    md_text = Path(input_file).read_text(encoding='utf-8')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    llm, _ = load_llm_config()
    messages = [
        ChatMessage(role='system', content=summary_query),
        ChatMessage(role='user', content=f'Please summarize the abstracts: {md_text}')
    ]
    response = llm.chat(messages)
    # save
    with open(os.path.join(output_dir, 'abstract_summary.md'), 'w') as f:
        f.write(response.message.content)
    print(f"Summaries saved to {output_dir}/abstract_summary.md")


def summarize_papers(input_file, output_dir, system_query):
    """
    Summarize papers
    :param input_file: the csv file containing the catalog
    :param output_dir: the directory to save the summaries
    :param system_query: the query to summarize the papers
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # turn the csv to markdown
    catalog = pd.read_csv(input_file)
    markdown_path = _turn_metadata_to_md(catalog, output_dir)
    summarize_abstracts(markdown_path, output_dir, system_query)

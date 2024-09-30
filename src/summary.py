import os

from llama_index.core.composability import QASummaryQueryEngineBuilder
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import FlatReader

from pathlib import Path
from src.utils import load_llm_config
import pandas as pd



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

def create_query_engine_from_abstracts(documents):
    """
    Create a query engine
    :param documents: the documents to create the query engine
    :return: the query engine
    """
    parser = MarkdownNodeParser()
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    transformation = [parser, splitter]
    query_engine_builder = QASummaryQueryEngineBuilder(transformations=transformation)
    query_engine = query_engine_builder.build_from_documents(documents)
    # save the query engine



def summarize_abstracts(input_file, output_dir, summary_query):
    """
    Summarize all the abstracts with the summary_query
    :param input_file: the markdown file containing the metadata
    :param output_dir: the directory to save the summaries
    :param summary_query: the query to summarize the abstracts
    :return: None
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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


def summarize_papers(mode, input_file=None, input_dir=None, output_dir=None, summary_query=None):
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
        summarize_abstracts(markdown_path, output_dir, summary_query)
    elif mode == 'document':
        summary_each_document(input_dir, summary_query)
    else:
        raise ValueError(f"Invalid mode: {mode}")
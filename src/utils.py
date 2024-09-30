import yaml
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from PyPDF2 import PdfReader
import pandas as pd

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_date(date_str):
    if date_str is None:
        date = datetime.now().date() - timedelta(days=3)
    else:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    return date

def load_llm_config():
    load_dotenv()
    llm = AzureOpenAI(
        model_name='gpt-4o',
        engine='gpt-4o',
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('GPT_API_VERSION'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )
    embed_model = AzureOpenAIEmbedding(
        model='text-embedding-ada-002',
        engine='text-embedding-ada-002',
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version='2024-02-01'
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

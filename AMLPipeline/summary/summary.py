from src.arxiv_crawler import search_papers, create_paper_catalog
from src.utils import load_config
from src.summary import summarize_papers
from datetime import datetime
import os
import argparse

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input', help='The input directory')
    parser.add_argument('--output_dir', type=str, default='output', help='The output directory')
    return parser.parse_args()

def main(args):
    system_query = """
    You are an expert in natural language processing and language models. Your task is to analyze a collection of paper abstracts in this field, provided in a markdown file containing the paper title, authors, abstract, and PDF URL for each paper. Instead of summarizing individual abstracts, focus on synthesizing information across all papers to provide a comprehensive overview of the research landscape.
   Perform the following tasks:
   
   Paper Catalog:
   List papers' published date range which you have analyzed.
    
    Identify Key Themes:
    
    List and briefly describe 3-5 major research themes or focus areas that emerge from the abstracts.
    For each theme, mention 2-3 representative papers (include title and URL).
  
    Innovative or High-Impact Papers:
    
    Identify 3-5 papers that appear to be the most innovative or impactful.
    For each, provide:
    a) Paper title and URL
    b) A one-sentence explanation of its key innovation or potential impact
    c) How it relates to or advances one of the key themes
      
    Research Trends Analysis:
    
    Describe 3-5 significant trends or shifts in research focus observed across the abstracts.
    Support each trend with examples from relevant papers.
    
    Methodological Approaches:
    
    Identify 3-5 common or emerging methodological approaches in the field.
    Briefly explain each approach and list 2-3 papers that exemplify it.
    
    Interdisciplinary Connections:
    
    Highlight any notable connections or applications to other fields of study mentioned in the abstracts.
    
    Concluding Overview:
    
    Provide a brief (3-5 sentences) high-level summary of the current state and direction of research in language models, based on your analysis of these abstracts.
    
    Remember to focus on synthesizing information across all abstracts rather than summarizing individual papers. Your analysis should provide insights into the collective body of research represented by these abstracts.
    """

    print('Summarizing abstracts...')
    date = datetime.now().strftime('%Y-%m-%d')
    catalog_path = os.path.join(args.input_dir, 'catalog.csv')
    summarize_papers(input_file=catalog_path, output_dir=args.output_dir, system_query=system_query)

if __name__ == '__main__':
    args = parse_parser()
    main(args)
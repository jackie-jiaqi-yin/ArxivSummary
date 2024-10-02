import sys
import os
# Get the directory three levels up
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append it to sys.path
sys.path.append(parent_dir)

import argparse
from src.arxiv_crawler import search_papers, create_paper_catalog
from datetime import datetime

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output', help='The output directory')
    return parser.parse_args()

def main(args):
    query = '(cat:cs.CL OR cat:cs.AI) AND (ti:"large language model" OR abs:"large language model" OR ti:LLM OR abs:LLM)'
    latest_num_papers = 100
    pdf_download =  False
    papers = search_papers(query, latest_num_papers)
    print(f"Found {len(papers)} papers")
    # download
    date = datetime.now().strftime('%Y-%m-%d')
    download_dir = os.path.join(args.output_dir, date)
    create_paper_catalog(papers, download_dir, pdf_download)


if __name__ == '__main__':
    args = parse_parser()
    main(args)
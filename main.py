from src.arxiv_crawler import search_papers, create_paper_catalog
from src.utils import load_config
from src.summary import summarize_papers
from datetime import datetime
import os
import argparse

def main(config):

    # arxiv crawl
    if config['arxiv_crawl']['run']:
        arxiv_config = config['arxiv_crawl']
        query = arxiv_config['query']
        max_results = arxiv_config.get('latest_num_papers', 100)
        pdf_download = arxiv_config.get('pdf_download', False)
        # find papers
        papers = search_papers(query, max_results)
        print(f"Found {len(papers)} papers")
        # download
        date = datetime.now().strftime('%Y-%m-%d')
        download_dir = os.path.join(config['arxiv_crawl']['output_dir'], date)
        create_paper_catalog(papers, download_dir, pdf_download)

    # summary papers
    if config['summary']['run']:
        summary_config = config['summary']
        # summarize all the abstracts
        print('Summarizing abstracts...')
        date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join(summary_config['output_dir'], date)
        catalog_path = os.path.join(output_dir, 'catalog.csv')
        summarize_papers(input_file=catalog_path, output_dir=output_dir, system_query=summary_config['system_query'])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Crawl arXiv papers and summarize them.')
    args.add_argument('--config', '-c', type=str, default='config.yaml', help='The config file path')
    args = args.parse_args()
    config = load_config(args.config)
    main(config)


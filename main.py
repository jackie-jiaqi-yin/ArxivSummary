from src.arxiv_crawler import search_papers, create_paper_catalog
from src.utils import load_config, format_date
from src.summary import summarize_papers
import os
import argparse

def main(config):

    # arxiv crawl
    if config['arxiv_crawl']['run']:
        arxiv_config = config['arxiv_crawl']
        query = arxiv_config['query']
        start_date = format_date(arxiv_config.get('start_date', None))
        end_date = format_date(arxiv_config.get('end_date', None))
        max_results = arxiv_config.get('max_results', 100)
        pdf_download = arxiv_config.get('pdf_download', False)
        # find papers
        papers = search_papers(query, start_date, end_date, max_results)
        print(f"Found {len(papers)} papers from {start_date} to {end_date}")
        # download
        download_dir = os.path.join(config['arxiv_crawl']['output_dir'], start_date.strftime('%Y-%m-%d'))
        create_paper_catalog(papers, download_dir, pdf_download)

    # summary papers
    if config['summary']['run']:
        summary_config = config['summary']
        # summarize all the abstracts
        print('Summarizing abstracts...')
        catalog_path = os.path.join(summary_config['input_dir'], 'catalog.csv')
        summarize_papers(input_file=catalog_path, output_dir=summary_config['output_dir'], system_query=summary_config['system_query'])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Crawl arXiv papers and summarize them.')
    args.add_argument('--config', '-c', type=str, default='config.yaml', help='The config file path')
    args = args.parse_args()
    config = load_config(args.config)
    main(config)


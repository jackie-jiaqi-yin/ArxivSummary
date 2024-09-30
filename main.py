from src.arxiv_crawler import search_papers, download_papers
from src.utils import load_config, format_date
from src.summary import summarize_papers
import os


def main(config_path):
    config = load_config(config_path)

    # arxiv crawl
    if config['arxiv_crawl']['run']:
        arxiv_config = config['arxiv_crawl']
        query = arxiv_config['query']
        start_date = format_date(arxiv_config.get('start_date', None))
        end_date = format_date(arxiv_config.get('end_date', None))
        max_results = arxiv_config.get('max_results', 100)
        # find papers
        papers = search_papers(query, start_date, end_date, max_results)
        print(f"Found {len(papers)} papers from {start_date} to {end_date}")
        # download
        download_dir = os.path.join(config['arxiv_crawl']['output_dir'], start_date.strftime('%Y-%m-%d'))
        download_papers(papers, download_dir)

    # summary papers
    if config['summary']['run']:
        summary_config = config['summary']
        if summary_config['mode'] == 'abstract': # summarize all the abstracts
            print('Summarizing abstracts...')
            metadata_path = os.path.join(summary_config['input_dir'], 'metadata.csv')
            summarize_papers('abstract', input_file=metadata_path, output_dir=summary_config['output_dir'], system_query=summary_config['system_query'])



if __name__ == '__main__':
    main('config.yml')


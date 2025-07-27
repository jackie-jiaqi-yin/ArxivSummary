import sys
import os
# Get the directory three levels up
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append it to sys.path
sys.path.append(parent_dir)

import argparse
from src.arxiv_crawler import ArxivCrawler
from src.utils.config import ArxivConfig


def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output', help='The output directory')
    parser.add_argument('--max_results', type=int, default=200, help='the max of papers to pull from ArXiv')
    parser.add_argument('--query', type=str, default='(cat:cs.CL OR cat:cs.AI) AND (ti:"large language model" OR abs:"large language model" OR ti:LLM OR abs:LLM)', help='ArXiv search query')
    parser.add_argument('--pdf_download', type=str, default='false', help='Whether to download PDFs')
    return parser.parse_args()

def main(args):
    # Convert string boolean to actual boolean
    pdf_download = args.pdf_download.lower() == 'true'
    
    # Create ArxivConfig with the parameters
    config = ArxivConfig(
        run=True,
        query=args.query,
        latest_num_papers=args.max_results,
        pdf_download=pdf_download,
        output_dir=args.output_dir
    )
    
    # Create ArxivCrawler instance and run
    crawler = ArxivCrawler(config)
    crawler.run(args.output_dir)


if __name__ == '__main__':
    args = parse_parser()
    main(args)
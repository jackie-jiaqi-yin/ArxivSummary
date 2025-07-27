import sys
import os
# Get the directory three levels up
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append it to sys.path
sys.path.append(parent_dir)

import argparse
from src.utils.config import SummaryConfig
from src.summarizer import PaperSummarizer
from src.html_generator import HTMLGenerator

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input', help='The input directory')
    parser.add_argument('--output_dir', type=str, default='output', help='The output directory')
    parser.add_argument('--system_query', type=str, default='You are an AI research assistant. Analyze the following research paper and provide a comprehensive summary.', help='System query for summarization')
    parser.add_argument('--batch_system_query', type=str, default='You are an AI research assistant. Analyze the following research papers and provide comprehensive summaries. Each summary should include the paper title, URL, key contributions, methodology, and main findings.', help='Batch system query for summarization')
    parser.add_argument('--auth_method', type=str, default='use_azure_ad', help='Authentication method: use_azure_ad, use_key, use_mi')
    parser.add_argument('--mi_client_id', type=str, default=None, help='Managed Identity client ID (required for use_mi)')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Model name to use')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max_concurrent_batches', type=int, default=3, help='Maximum concurrent batches')
    return parser.parse_args()

def main(args):
    # Create SummaryConfig with all required parameters
    config = SummaryConfig(
        run=True,
        system_query=args.system_query,
        batch_system_query=args.batch_system_query,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        auth_method=args.auth_method,
        mi_client_id=args.mi_client_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_concurrent_batches=args.max_concurrent_batches
    )
    
    # Create PaperSummarizer instance
    summarizer = PaperSummarizer(config)
    
    # Find catalog.csv in input directory
    catalog_path = os.path.join(args.input_dir, 'catalog.csv')
    
    # Run summarizer
    print('Starting paper summarization...')
    summarizer.run(catalog_path, args.output_dir, args.model_name)
    
    # HTML Generation
    print('Starting HTML generation...')
    html_generator = HTMLGenerator()
    
    markdown_file = os.path.join(args.output_dir, 'abstract_summary.md')
    html_file = html_generator.run(markdown_file, args.output_dir)
    
    print(f'Successfully generated HTML: {html_file}')

if __name__ == '__main__':
    args = parse_parser()
    main(args)
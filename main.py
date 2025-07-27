import os
import argparse
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables will only be loaded from system.")
    print("To install: pip install python-dotenv")

from src.utils.config import Config
from src.arxiv_crawler import ArxivCrawler
from src.summarizer import PaperSummarizer
from src.html_generator import HTMLGenerator


def main(config: Config):
    """
    Main orchestration function for the ArXiv summary pipeline
    
    Args:
        config (Config): Configuration object
    """
    
    # Step 1: ArXiv Paper Crawling
    if config.arxiv_crawl.run:
        print("=== Starting ArXiv Paper Crawling ===")
        crawler = ArxivCrawler(config.arxiv_crawl)
        
        # Create date-based directory
        date = datetime.now().strftime('%Y-%m-%d')
        download_dir = os.path.join(config.arxiv_crawl.output_dir, date)
        
        # Run crawler
        catalog_df = crawler.run(download_dir)
        
        if catalog_df is not None:
            print(f"Successfully crawled {len(catalog_df)} papers")
        else:
            print("No papers found, exiting...")
            return
    
    # Step 2: Paper Summarization
    if config.summary.run:
        print("\n=== Starting Paper Summarization ===")
        print(f"Using model: {config.summary.model_name}")
        print(f"Batch size: {config.summary.batch_size}")
        print(f"Max concurrent batches: {config.summary.max_concurrent_batches}")
        
        summarizer = PaperSummarizer(config.summary)
        
        # Create date-based directory
        date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join(config.summary.output_dir, date)
        catalog_path = os.path.join(output_dir, 'catalog.csv')
        
        # Run summarizer
        summarizer.run(catalog_path, output_dir, config.summary.model_name)
        
        # Step 3: HTML Generation
        print("\n=== Starting HTML Generation ===")
        html_generator = HTMLGenerator()
        
        markdown_file = os.path.join(output_dir, 'abstract_summary.md')
        html_file = html_generator.run(markdown_file, output_dir)
        
        print(f"Successfully generated HTML: {html_file}")
    
    print("\n=== Pipeline Complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl arXiv papers and generate HTML summary.')
    parser.add_argument('--config', '-c', type=str, default='config.yml', 
                       help='The config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    
    # Run main pipeline
    main(config)
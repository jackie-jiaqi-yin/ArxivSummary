# ArxivSummary

ArxivSummary is a Python project designed to crawl arXiv papers, create a catalog of papers, and generate summaries using advanced language models.

## Features

- Search for papers from arXiv based on specified criteria
- Create a catalog of papers with metadata
- Summarize paper abstracts using AI-powered language models
- Configurable settings via YAML configuration file

## Prerequisites

- Python 3.10
- Conda (for environment management)
- Azure OpenAI account with access to GPT models. Or you can use other LM services

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/placeboo/ArxivSummary.git
   cd ArxivSummary
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate ArxivSummary
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your Azure OpenAI credentials:
   ```
   MODEL_NAME=your_model_name
   ENGINE=your_engine
   AZURE_OPENAI_ENDPOINT=your_endpoint
   GPT_API_VERSION=your_api_version
   AZURE_OPENAI_API_KEY=your_api_key
   ```

## Usage

1. Configure your search and summary settings in `config.yml`. :
   ```yaml
   arxiv_crawl:
     run: True
     pdf_download: False
     query: '(cat:cs.CL OR cat:cs.AI) AND (ti:"large language model" OR abs:"large language model" OR ti:LLM OR abs:LLM)'
     latest_num_papers: 200
     output_dir: 'data/arxiv'

   summary:
     run: True
     input_dir: 'data/arxiv/'
     output_dir: 'data/arxiv/'
     system_query: "Your system query here..."
   ```
   ### Configuration Schema Explanation

   #### arxiv_crawl
   - `run` (boolean): Set to `True` to enable arXiv crawling, `False` to skip this step.
   - `pdf_download` (boolean): Set to `True` to download PDF files of papers, `False` to skip PDF downloads.
   - `query` (string): The arXiv query string to filter papers. You can use arXiv categories (e.g., 'cat:cs.CL') and search terms in titles or abstracts.  For advanced query syntax documentation, see the arXiv API [User Manual](https://arxiv.org/help/api/user-manual#query_detail)

   - `latest_num_papers` (integer): The maximum number of papers to retrieve from the most recent submissions.
   - `output_dir` (string): The directory where the crawled data will be saved.

   #### summary
   - `run` (boolean): Set to `True` to enable summary generation, `False` to skip this step.
   - `input_dir` (string): The directory containing the input data (should match the `output_dir` from `arxiv_crawl`).
   - `output_dir` (string): The directory where the generated summaries will be saved.
   - `system_query` (string): The prompt or instructions for the AI model to generate the summary. This should be a detailed description of how to analyze and summarize the papers.


2. Run the main script:
   ```
   python main.py -c config.yml
   ```

## Project Structure

- `main.py`: Entry point of the application
- `src/`:
  - `arxiv_crawler.py`: Functions for searching arXiv papers
  - `summary.py`: Functions for summarizing paper abstracts
  - `utils.py`: Utility functions for configuration, date formatting, and LLM setup
- `config.yml`: Configuration file for search and summary settings
- `environment.yml`: Conda environment specification

## Output

- A CSV catalog of papers will be saved in the specified output directory, e.g. [catalog.csv](data%2Farxiv%2F2024-09-30%2Fcatalog.csv)
- A markdown file containing the metadata of papers will be generated in the output directory, e.g. [catalog.md](data%2Farxiv%2F2024-09-30%2Fcatalog.md)
- A markdown file containing the summary of paper abstracts will be generated in the output directory, e.g. [abstract_summary.md](data%2Farxiv%2F2024-09-30%2Fabstract_summary.md)

## Summary Generation

The system uses a sophisticated prompt to analyze and summarize the collection of paper abstracts. The summary includes:

- Paper catalog with publication date range
- Key research themes with representative papers
- Innovative or high-impact papers
- Research trends analysis
- Common methodological approaches
- Interdisciplinary connections
- A concluding overview of the current state and direction of research

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
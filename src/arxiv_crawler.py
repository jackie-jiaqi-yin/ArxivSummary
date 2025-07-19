import arxiv
import pandas as pd
import requests
import os
from typing import List
from src.utils.config import ArxivConfig
from src.utils.helpers import create_saved_title


class ArxivCrawler:
    def __init__(self, config: ArxivConfig):
        self.config = config
        self.client = arxiv.Client()
    
    def search_papers(self) -> List[arxiv.Result]:
        """
        Search arXiv papers using the configuration
        
        Returns:
            List[arxiv.Result]: A list of arXiv papers
        """
        search = arxiv.Search(
            query=self.config.query,
            max_results=self.config.latest_num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        results = self.client.results(search)
        return list(results)
    
    def create_paper_catalog(self, papers: List[arxiv.Result], save_dir: str) -> pd.DataFrame:
        """
        Create a catalog of papers and optionally download PDFs
        
        Args:
            papers (List[arxiv.Result]): List of papers
            save_dir (str): Directory to save the catalog and PDFs
            
        Returns:
            pd.DataFrame: DataFrame with paper information
        """
        if len(papers) == 0:
            print('No papers found')
            return None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        catalog = {
            'title': [],
            'authors': [],
            'abstract': [],
            'published': [],
            'category': [],
            'pdf_url': [],
            'saved_title': []
        }
        
        # Create catalog
        for paper in papers:
            catalog['title'].append(paper.title)
            authors = ', '.join(author.name for author in paper.authors)
            catalog['authors'].append(authors)
            catalog['abstract'].append(paper.summary)
            catalog['published'].append(paper.published)
            catalog['category'].append(paper.primary_category)
            catalog['pdf_url'].append(paper.pdf_url)
            save_title = create_saved_title(paper.title)
            catalog['saved_title'].append(save_title)
        
        df = pd.DataFrame(catalog)
        catalog_path = os.path.join(save_dir, 'catalog.csv')
        df.to_csv(catalog_path, index=False)
        print(f"Saved {df.shape[0]} papers to {catalog_path}")
        
        # Download PDFs if requested
        if self.config.pdf_download:
            self._download_pdfs(df, save_dir)
        
        return df
    
    def _download_pdfs(self, df: pd.DataFrame, save_dir: str) -> None:
        """
        Download PDFs for papers in the catalog
        
        Args:
            df (pd.DataFrame): DataFrame with paper information
            save_dir (str): Directory to save PDFs
        """
        pdf_count = 0
        for i, row in df.iterrows():
            title = row['saved_title']
            pdf_url = row['pdf_url']
            try:
                response = requests.get(pdf_url)
                pdf_path = os.path.join(save_dir, f'{title}.pdf')
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                pdf_count += 1
            except Exception as e:
                print(f"Failed to download {title}: {e}")
        
        print(f"Downloaded {pdf_count} papers")
    
    def run(self, save_dir: str) -> pd.DataFrame:
        """
        Run the complete ArXiv crawling process
        
        Args:
            save_dir (str): Directory to save results
            
        Returns:
            pd.DataFrame: DataFrame with paper information
        """
        papers = self.search_papers()
        print(f"Found {len(papers)} papers")
        
        return self.create_paper_catalog(papers, save_dir)
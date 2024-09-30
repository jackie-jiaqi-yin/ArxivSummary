import arxiv
import pandas as pd
import requests
import os
from src.utils import create_saved_title

def search_papers(query, start_date, end_date, max_results=100):
    """
    Search arXiv papers using arxiv API and return a list of papers
    :param query: the query string to filter the papers
    :param start_date: search for papers published after this date (inclusive)
    :param end_date: search for papers published before this date (inclusive)
    :param max_results: the maximum number of urls from api
    :return: a list of papers
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query = query,
        max_results=max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending,
    )

    # filter papers based on date
    results = []
    for paper in client.results(search):
        if start_date <= paper.published.date() <= end_date:
            results.append(paper)
        elif paper.published.date() < start_date:
            break
    return results

def create_paper_catalog(papers, save_dir, download_pdf=False):
    """
    Create a catalog of papers and download the pdfs
    :param papers: a list of papers
    :param save_dir: the directory to save the pdfs
    :param download_pdf: whether to download the pdfs
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
        'pdf_url': [],
        'saved_title': []
    }
    
    # create catalog
    for paper in papers:
        catalog['title'].append(paper.title)
        authors = ', '.join(author.name for author in paper.authors)
        catalog['authors'].append(authors)
        catalog['abstract'].append(paper.summary)
        catalog['published'].append(paper.published)
        catalog['pdf_url'].append(paper.pdf_url)
        save_title = create_saved_title(paper.title)
        catalog['saved_title'].append(save_title)
    df = pd.DataFrame(catalog)
    df.to_csv(f'{save_dir}/catalog.csv', index=False)
    print(f"Saved {df.shape[0]} papers to {save_dir}/catalog.csv")
    if download_pdf:
        pdf_count = 0
        for i, row in df.iterrows():
            title = row['saved_title']
            pdf_url = row['pdf_url']
            try:
                response = requests.get(pdf_url)
                with open(f'{save_dir}/{title}.pdf', 'wb') as f:
                    f.write(response.content)
                pdf_count += 1
            except Exception as e:
                print(f"Failed to download {title}")
        print(f"Downloaded {pdf_count} papers")

    return df


import arxiv
import pandas as pd
from datetime import datetime, timedelta
import requests
import io
import os

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

def download_papers(papers, save_dir):
    """
    Download the papers as pdf files and save a csv file with metadata
    :param papers: the results from search_papers
    :param save_dir: the directory to save the papers
    :return: metadata dataframe
    """
    if len(papers) == 0:
        print('No papers found')
        return None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    metadata = {
        'title': [],
        'authors': [],
        'abstract': [],
        'published': [],
        'pdf_url': [],
        'saved_title': []
    }
    for paper in papers:

        # download pdf
        # remove the special charaters and spaces from the title
        title = paper.title.replace(' ', '_').replace('/', '_')
        try:
            paper.download_pdf(dirpath=save_dir, filename=f"{title}.pdf")
            metadata['title'].append(paper.title)
            authors = ', '.join(author.name for author in paper.authors)
            metadata['authors'].append(authors)
            metadata['abstract'].append(paper.summary)
            metadata['published'].append(paper.published)
            metadata['pdf_url'].append(paper.pdf_url)
            metadata['saved_title'].append(title)
        except Exception as e:
            print(f"Failed to download {paper.title}")
    df = pd.DataFrame(metadata)
    print(f"Downloaded {df.shape[0]} papers")
    df.to_csv(f'{save_dir}/metadata.csv', index=False)
    return df


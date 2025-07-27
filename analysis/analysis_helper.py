import pandas as pd
from typing import Dict, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

def collect_history_summary(input_dir: str) -> Dict:
    """
    collect the history from the input directory. There subdirectories in the input directory, date as name.
    Each subdirectory contains:
    - abstract_summary.md: llm summary of the abstracts
    - catalog.csv: the papers metadata
    :param input_dir: str
    :return: {date: {abstract_summary: str, catalog: pd.DataFrame}}
    """
    results = {}
    for date in os.listdir(input_dir):
        date_dir = os.path.join(input_dir, date)
        if os.path.isdir(date_dir):
            abstract_summary_filename = os.path.join(date_dir, 'abstract_summary.md')
            catalog_filename = os.path.join(date_dir, 'catalog.csv')

            results[date] = {
                'abstract_summary': open(abstract_summary_filename, 'r').read(),
                'catalog': pd.read_csv(catalog_filename)
            }
    return results


def collect_paper_log(data: Dict) -> pd.DataFrame:
    """
    organize the catalog from the collected history
    :param data: the output from collect_history_summary
    :return: the data frame containing the unique papers
    """
    all_paper_log = pd.concat([data[date]['catalog'] for date in data])
    all_paper_log = all_paper_log.drop_duplicates(subset=['pdf_url'], keep='first')
    return all_paper_log


def extract_header2(text: str) -> List[str]:
    """
    extract all the header2 from the text
    :param text: str
    :return: [Research Search, ...]
        Example:
        text = '''
        ## **2. Key Research Themes**
        ### 1. First Theme
        ### 2. Second Theme
        ## Next Section
        '''
        get_header3_under_header2(text, "Key Research Themes")
        ['First Theme', 'Second Theme']
    """
    h2_pattern = r'^(?<!#)##(?!#)\s*(?:\d+[\.:]?\s*)?([^\n]+)'
    headers = re.finditer(h2_pattern, text, re.MULTILINE)
    # Clean up the matches by removing ** and numbers
    return [re.sub(r'^\*\*\d+\.?\s*|\*\*', '', match.group(1)).strip() for match in headers]


def get_header3_under_header2(text: str, header2: str) -> List[str]:
    """
    Get all header3 under a specific header2, with debug prints
    """
    h2_pattern = r'^(?<!#)##(?!#)\s*(?:\d+[\.:]?\s*)?'

    # split the text into sections by h2 hearders
    sections = re.split(h2_pattern, text, flags=re.MULTILINE)

    # find the section that contains the header2
    target_section = None
    for section in sections:
        # clean up the section start to handle the potential formating
        clean_section = re.sub(r'^\*\*\d+\.?\s*|\*\*', '', section.strip())
        if clean_section.startswith(header2):
            target_section = section
            break

    if not target_section:
        print(f"Header2 '{header2}' not found")
        return []

    # find h3 headers under the target section
    h3_pattern = r'^(?<!#)###(?!#)\s*(?:\d+[\.:]?\s*)?([^\n]+)'
    h3_matches = re.finditer(h3_pattern, target_section, re.MULTILINE)
    # Clean up the headers
    cleaned_headers = []
    for match in h3_matches:
        header = match.group(1).strip()
        # If there's a colon, keep only what's after it
        if ':' in header:
            header = header.split(':', 1)[1]
        else:
            # If no colon, just remove any numbering at the start
            header = re.sub(r'^\s*(?:\d+\.|\d+\)|\(\d+\)|\d+)\s*', '', header)
        # Remove any asterisks
        header = re.sub(r'^\*\*\d+\.?\s*|\*\*', '', header)
        # Remove leading/trailing whitespace
        header = header.strip()
        if header:  # Only add non-empty headers
            cleaned_headers.append(header)

    return cleaned_headers


def list_all_header2(data: Dict) -> Dict[str, int]:
    """
    list all the header2 from the abstract summaries
    :param data: the output from collect_history_summary
    :return: a dictionary of header2 and the count
    """
    results = {}
    for date in data:
        abstract_summary = data[date]['abstract_summary']
        headers = extract_header2(abstract_summary)
        for header in headers:
            results[header] = results.get(header, 0) + 1
    return results


def list_all_header3(data: Dict, header2: str) -> Dict[str, List[str]]:
    """
    list all the header3 under a specific header2 from the abstract summaries
    :param data: the output from collect_history_summary
    :param header2: str
    :return: a dictionary of header3 and the count
    """
    results = {}
    for date in data:
        abstract_summary = data[date]['abstract_summary']
        header3s = get_header3_under_header2(abstract_summary, header2)
        results[date] = header3s
    return results



# Plotting ---------------------------------------------------------------------


def set_plot_style(fig_size=(10, 6), style='whitegrid', font_scale=1.2):
    """
    Configure matplotlib plot style for presentations.

    Parameters:
    -----------
    fig_size : tuple, default=(10, 6)
        Figure size in inches (width, height)
    style : str, default='whitegrid'
        Seaborn style ('whitegrid', 'white', 'dark', 'darkgrid', 'ticks')
    font_scale : float, default=1.2
        Scale factor for font sizes

    Returns:
    --------
    dict : Dictionary containing the style parameters for reset if needed
    """
    # Store original settings
    original_settings = {
        'font.family': plt.rcParams['font.family'],
        'font.size': plt.rcParams['font.size'],
        'figure.figsize': plt.rcParams['figure.figsize']
    }

    # Set the style using seaborn
    sns.set_style(style)
    sns.set_context("talk", font_scale=font_scale)

    # Set figure size
    plt.rcParams['figure.figsize'] = fig_size

    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

    # Grid settings
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5

    # Line settings
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

    # Axes settings
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Tick settings
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Legend settings
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.edgecolor'] = '0.8'

    return original_settings

def plot_daily_paper_count(data: pd.DataFrame):
    """
    Plot the daily paper count
    :param data: the data from collect_paper_log
    """
    data['date'] = pd.to_datetime(data['published']).dt.date
    daily_count = data.groupby('date').size().rename('count')
    # patch 0 for missing dates
    daily_count = daily_count.asfreq('D', fill_value=0).reset_index()
    # draw bar plot
    set_plot_style()
    sns.barplot(data=daily_count, x='date', y='count', color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Paper Count')
    plt.xlabel('Published Date')
    plt.title('Daily Paper Count')
    plt.tight_layout()
    plt.show()
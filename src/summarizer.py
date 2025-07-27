import os
import time
import pandas as pd
from pathlib import Path
from typing import Optional, List
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import math
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re

from src.utils.config import SummaryConfig
from src.utils.llm_service import get_llm


class PaperSummarizer:
    def __init__(self, config: SummaryConfig):
        self.config = config
        self.papers_per_batch = config.batch_size
        
    def _get_date_range(self, catalog: pd.DataFrame) -> str:
        """
        Calculate date range from catalog DataFrame
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information
            
        Returns:
            str: Date range string (e.g., "2024-01-15 to 2024-01-20")
        """
        # Convert published dates to datetime if they're not already
        dates = []
        for _, row in catalog.iterrows():
            date_str = row['published']
            if hasattr(date_str, 'strftime'):
                dates.append(date_str)
            else:
                # Parse string date
                try:
                    # Handle different date formats
                    if 'T' in str(date_str):
                        # ISO format with time
                        date_obj = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
                    else:
                        # Simple date format
                        date_obj = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')
                    dates.append(date_obj)
                except:
                    # Fallback: use string as-is
                    dates.append(str(date_str)[:10])
        
        if not dates:
            return "No dates available"
        
        # Get min and max dates
        min_date = min(dates)
        max_date = max(dates)
        
        # Format as strings
        if hasattr(min_date, 'strftime'):
            min_str = min_date.strftime('%Y-%m-%d')
        else:
            min_str = str(min_date)
            
        if hasattr(max_date, 'strftime'):
            max_str = max_date.strftime('%Y-%m-%d')
        else:
            max_str = str(max_date)
        
        if min_str == max_str:
            return min_str
        else:
            return f"{min_str} to {max_str}"
    
    def _create_markdown_from_catalog(self, catalog: pd.DataFrame, output_dir: str) -> str:
        """
        Convert catalog DataFrame to markdown format
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information
            output_dir (str): Directory to save markdown file
            
        Returns:
            str: Path to the created markdown file
        """
        output_file = os.path.join(output_dir, 'catalog.md')
        
        with open(output_file, encoding='utf-8', mode='w') as f:
            for _, row in catalog.iterrows():
                title = row['title']
                authors = row['authors']
                abstract = row['abstract']
                url = row['pdf_url']
                date = row['published'].strftime('%Y-%m-%d') if hasattr(row['published'], 'strftime') else str(row['published'])[:10]
                
                f.write(f"## {title}\n\n")
                f.write(f"**Authors**: {authors}\n\n")
                f.write(f"**Abstract**: {abstract}\n\n")
                f.write(f"**URL**: {url}\n\n")
                f.write(f"**Published**: {date}\n\n")
        
        print(f"Catalog saved to {output_file}")
        return output_file
    
    def _get_llm_instance(self, model_name: str = 'o1'):
        """Get LLM instance based on configuration"""
        if self.config.auth_method == 'use_azure_ad':
            return get_llm(
                model_name=model_name,
                auth_method='use_azure_ad'
            )
        elif self.config.auth_method == 'use_key':
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is required when using 'use_key' auth method")
            return get_llm(
                model_name=model_name,
                auth_method='use_key',
                api_key=api_key
            )
        elif self.config.auth_method == 'use_mi':
            return get_llm(
                model_name=model_name,
                auth_method='use_mi',
                mi_client_id=self.config.mi_client_id
            )
        else:
            raise ValueError(f"Unsupported auth_method: {self.config.auth_method}")
    
    def _summarize_batch(self, papers_text: str, batch_num: int, model_name: str = 'o1') -> str:
        """
        Summarize a batch of papers (synchronous version)
        
        Args:
            papers_text (str): Markdown text of papers
            batch_num (int): Batch number for identification
            model_name (str): Name of the LLM model to use
            
        Returns:
            str: Summary of the batch
        """
        llm = self._get_llm_instance(model_name)
        
        messages = [
            ChatMessage(role='system', content=self.config.batch_system_query),
            ChatMessage(role='user', content=f'Please summarize this batch of papers: {papers_text}')
        ]
        
        # Retry logic for LLM calls
        attempts = 0
        while attempts <= 3:
            try:
                response = llm.chat(messages)
                print(f"Completed batch {batch_num} summary")
                return response.message.content
                
            except Exception as e:
                attempts += 1
                if attempts > 3:
                    print(f"Failed to summarize batch {batch_num}: {str(e)}")
                    raise
                print(f"Retrying batch {batch_num}...{attempts}")
                time.sleep(10)
    
    async def _summarize_batch_async(self, papers_text: str, batch_num: int, model_name: str = 'o1') -> str:
        """
        Summarize a batch of papers asynchronously
        
        Args:
            papers_text (str): Markdown text of papers
            batch_num (int): Batch number for identification
            model_name (str): Name of the LLM model to use
            
        Returns:
            str: Summary of the batch
        """
        def _sync_summarize():
            return self._summarize_batch(papers_text, batch_num, model_name)
        
        # Run the synchronous LLM call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, _sync_summarize)
            return result
    
    def _parse_llm_response(self, response_text: str) -> str:
        """
        Parse LLM response to remove unnecessary headers and content before main sections
        
        Args:
            response_text (str): Raw LLM response
            
        Returns:
            str: Cleaned response starting from main content
        """
        # Look for main section headers to start from
        main_section_patterns = [
            r'## \*\*Paper Catalog\*\*',
            r'## \*\*Key Research Themes\*\*',
            r'## \*\*Methodological Approaches\*\*',
            r'## Paper Catalog',
            r'## Key Research Themes',
            r'## Methodological Approaches'
        ]
        
        for pattern in main_section_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                # Found a main section, start from there
                cleaned_response = response_text[match.start():]
                print(f"Cleaned response: removed {match.start()} characters from beginning")
                return cleaned_response
        
        # If no main sections found, return original (fallback)
        print("No main sections found, returning original response")
        return response_text
    
    def _combine_batch_summaries(self, batch_summaries: List[str], catalog: pd.DataFrame, model_name: str = 'o1') -> str:
        """
        Combine multiple batch summaries into final comprehensive summary
        
        Args:
            batch_summaries (List[str]): List of batch summaries
            catalog (pd.DataFrame): Original catalog for date range calculation
            model_name (str): Name of the LLM model to use
            
        Returns:
            str: Final combined summary
        """
        llm = self._get_llm_instance(model_name)
        
        combined_text = "\n\n---\n\n".join([f"## Batch {i+1} Summary:\n{summary}" for i, summary in enumerate(batch_summaries)])
        
        messages = [
            ChatMessage(role='system', content=self.config.system_query),
            ChatMessage(role='user', content=f'Please synthesize these batch summaries into a comprehensive final report: {combined_text}')
        ]
        
        # Retry logic for LLM calls
        attempts = 0
        while attempts <= 3:
            try:
                response = llm.chat(messages)
                print("Completed final summary combination")
                
                # Parse the response to remove unnecessary headers
                cleaned_response = self._parse_llm_response(response.message.content)
                
                # Add paper catalog section with date range at the beginning
                date_range = self._get_date_range(catalog)
                paper_catalog_section = f"""## **Paper Catalog**

**Date Range**: {date_range}

**Total Papers Analyzed**: {len(catalog)}

---

"""
                
                # Combine catalog section with cleaned response
                final_response = paper_catalog_section + cleaned_response
                
                return final_response
                
            except Exception as e:
                attempts += 1
                if attempts > 3:
                    print(f"Failed to combine batch summaries: {str(e)}")
                    raise
                print(f"Retrying final combination...{attempts}")
                time.sleep(10)
    
    def _recursive_summarize(self, catalog: pd.DataFrame, output_dir: str, model_name: str = 'o1') -> None:
        """
        Perform recursive summarization for large paper sets
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information
            output_dir (str): Directory to save summaries
            model_name (str): Name of the LLM model to use
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        total_papers = len(catalog)
        num_batches = math.ceil(total_papers / self.papers_per_batch)
        
        print(f"Processing {total_papers} papers in {num_batches} batches of {self.papers_per_batch} papers each")
        print("Using async processing for faster batch summarization...")
        
        # Run async batch processing
        batch_summaries = asyncio.run(self._process_batches_async(catalog, output_dir, model_name))
        
        # Combine all batch summaries
        print("Combining all batch summaries into final report...")
        final_summary = self._combine_batch_summaries(batch_summaries, catalog, model_name)
        
        # Save final summary
        summary_path = os.path.join(output_dir, 'abstract_summary.md')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        
        print(f"Final summary saved to {summary_path}")
    
    async def _process_batches_async(self, catalog: pd.DataFrame, output_dir: str, model_name: str = 'o1') -> List[str]:
        """
        Process all batches asynchronously
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information
            output_dir (str): Directory to save summaries
            model_name (str): Name of the LLM model to use
            
        Returns:
            List[str]: List of batch summaries
        """
        total_papers = len(catalog)
        num_batches = math.ceil(total_papers / self.papers_per_batch)
        
        # Create tasks for all batches
        tasks = []
        batch_data = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.papers_per_batch
            end_idx = min((batch_idx + 1) * self.papers_per_batch, total_papers)
            
            # Get batch of papers
            batch_catalog = catalog.iloc[start_idx:end_idx]
            
            # Create markdown for this batch
            batch_markdown = self._create_batch_markdown(batch_catalog)
            
            # Store batch data for later saving
            batch_data.append({
                'batch_idx': batch_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'batch_markdown': batch_markdown
            })
            
            # Create async task
            task = self._summarize_batch_async(batch_markdown, batch_idx + 1, model_name)
            tasks.append(task)
            
            print(f"Created task for batch {batch_idx + 1}/{num_batches} (papers {start_idx + 1}-{end_idx})")
        
        # Run all batch summarization tasks concurrently
        print(f"Starting concurrent processing of {len(tasks)} batches...")
        start_time = time.time()
        
        # Process batches in smaller concurrent groups to avoid overwhelming the API
        max_concurrent = min(self.config.max_concurrent_batches, len(tasks))
        batch_summaries = []
        
        for i in range(0, len(tasks), max_concurrent):
            batch_group = tasks[i:i + max_concurrent]
            batch_indices = list(range(i, min(i + max_concurrent, len(tasks))))
            
            print(f"Processing batch group {i//max_concurrent + 1} ({len(batch_group)} batches)")
            
            # Wait for this group to complete
            group_results = await asyncio.gather(*batch_group, return_exceptions=True)
            
            # Process results and save individual batch summaries
            for idx, result in enumerate(group_results):
                batch_idx = batch_indices[idx]
                
                if isinstance(result, Exception):
                    print(f"Error in batch {batch_idx + 1}: {result}")
                    # Use a fallback summary or re-raise
                    raise result
                else:
                    batch_summaries.append(result)
                    
                    # Save individual batch summary
                    batch_file = os.path.join(output_dir, f'batch_{batch_idx + 1}_summary.md')
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        f.write(result)
            
            # Add small delay between groups to be nice to the API
            if i + max_concurrent < len(tasks):
                await asyncio.sleep(1)
        
        end_time = time.time()
        print(f"Completed all {len(tasks)} batch summaries in {end_time - start_time:.2f} seconds")
        
        return batch_summaries
    
    def _create_batch_markdown(self, catalog: pd.DataFrame) -> str:
        """
        Create markdown text for a batch of papers
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information for this batch
            
        Returns:
            str: Markdown text for the batch
        """
        markdown_parts = []
        
        for _, row in catalog.iterrows():
            title = row['title']
            authors = row['authors']
            abstract = row['abstract']
            url = row['pdf_url']
            date = row['published'].strftime('%Y-%m-%d') if hasattr(row['published'], 'strftime') else str(row['published'])[:10]
            
            paper_md = f"""## {title}

**Authors**: {authors}

**Abstract**: {abstract}

**URL**: {url}

**Published**: {date}

"""
            markdown_parts.append(paper_md)
        
        return "\n".join(markdown_parts)
    
    def summarize_papers(self, input_file: str, output_dir: str, model_name: str = 'o1') -> None:
        """
        Complete paper summarization process
        
        Args:
            input_file (str): Path to CSV file with paper catalog
            output_dir (str): Directory to save results
            model_name (str): Name of the LLM model to use
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load catalog
        catalog = pd.read_csv(input_file)
        total_papers = len(catalog)
        
        print(f"Loaded {total_papers} papers for summarization")
        
        # Create full markdown catalog for reference
        markdown_path = self._create_markdown_from_catalog(catalog, output_dir)
        
        # Use recursive summarization for large paper sets
        if total_papers > self.papers_per_batch:
            print(f"Using recursive summarization (batch size: {self.papers_per_batch})")
            self._recursive_summarize(catalog, output_dir, model_name)
        else:
            print(f"Using direct summarization (total papers: {total_papers})")
            self._direct_summarize(catalog, output_dir, model_name)
    
    def _direct_summarize(self, catalog: pd.DataFrame, output_dir: str, model_name: str = 'o1') -> None:
        """
        Direct summarization for small paper sets
        
        Args:
            catalog (pd.DataFrame): DataFrame with paper information
            output_dir (str): Directory to save summary
            model_name (str): Name of the LLM model to use
        """
        # Create markdown for all papers
        markdown_text = self._create_batch_markdown(catalog)
        
        # Get LLM instance
        llm = self._get_llm_instance(model_name)
        
        messages = [
            ChatMessage(role='system', content=self.config.system_query),
            ChatMessage(role='user', content=f'Please summarize the abstracts: {markdown_text}')
        ]
        
        # Retry logic for LLM calls
        attempts = 0
        while attempts <= 3:
            try:
                response = llm.chat(messages)
                
                # Parse the response to remove unnecessary headers
                cleaned_response = self._parse_llm_response(response.message.content)
                
                # Add paper catalog section with date range at the beginning
                date_range = self._get_date_range(catalog)
                paper_catalog_section = f"""## **Paper Catalog**

**Date Range**: {date_range}

**Total Papers Analyzed**: {len(catalog)}

---

"""
                
                # Combine catalog section with cleaned response
                final_response = paper_catalog_section + cleaned_response
                
                # Save summary
                summary_path = os.path.join(output_dir, 'abstract_summary.md')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(final_response)
                
                print(f"Summary saved to {summary_path}")
                return
                
            except Exception as e:
                attempts += 1
                if attempts > 3:
                    print(f"Failed to summarize the abstracts: {str(e)}")
                    raise
                print(f"Retrying...{attempts}")
                time.sleep(10)
    
    def run(self, input_file: str, output_dir: str, model_name: str = 'o1') -> None:
        """
        Run the complete summarization process
        
        Args:
            input_file (str): Path to CSV file with paper catalog
            output_dir (str): Directory to save results
            model_name (str): Name of the LLM model to use
        """
        print('Summarizing abstracts...')
        self.summarize_papers(input_file, output_dir, model_name)
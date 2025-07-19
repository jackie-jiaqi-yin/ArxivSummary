import yaml
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArxivConfig:
    run: bool
    query: str
    latest_num_papers: int
    pdf_download: bool
    output_dir: str


@dataclass
class SummaryConfig:
    run: bool
    system_query: str
    batch_system_query: str
    input_dir: str
    output_dir: str
    auth_method: str  # 'use_azure_ad', 'use_key', 'use_mi'
    mi_client_id: Optional[str]  # Only required when auth_method is 'use_mi'
    model_name: str
    batch_size: int
    max_concurrent_batches: int


@dataclass
class Config:
    arxiv_crawl: ArxivConfig
    summary: SummaryConfig
    
    @classmethod
    def load(cls, config_path: str = "config.yml") -> 'Config':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        arxiv_config = ArxivConfig(**config_data['arxiv_crawl'])
        summary_config = SummaryConfig(**config_data['summary'])
        
        return cls(
            arxiv_crawl=arxiv_config,
            summary=summary_config
        )
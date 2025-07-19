"""
Azure configuration helper for local execution of AML Pipeline scripts.
This file loads Azure configuration from environment variables for test_* and submit_* scripts.
"""
import os
import json
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")


def get_azure_config() -> Dict[str, Any]:
    """
    Get Azure configuration from environment variables
    
    Returns:
        Dict[str, Any]: Configuration dictionary with Azure resource details
        
    Raises:
        ValueError: If required environment variables are not set
    """
    # Required environment variables
    config = {
        'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
        'resource_group': os.getenv('AZURE_RESOURCE_GROUP_NAME'),
        'workspace_name': os.getenv('AZURE_AML_WORKSPACE_NAME'),
    }
    
    # Check for missing required variables
    missing_vars = [var for var, value in config.items() if value is None]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars.upper())}")
    
    return config


def get_azure_ml_client():
    """
    Get an Azure ML client using environment variables
    
    Returns:
        MLClient: Configured Azure ML client
    """
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    
    config = get_azure_config()
    
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name']
    )


def get_datastore_path(path_suffix: str) -> str:
    """
    Get the full datastore path with environment variables
    
    Args:
        path_suffix (str): The suffix path (e.g., 'ArxivSummary/current')
        
    Returns:
        str: Full datastore path
    """
    config = get_azure_config()
    datastore_name = os.getenv('AZURE_DATASTORE_NAME', 'workspaceblobstore')
    
    return (f"azureml://subscriptions/{config['subscription_id']}/"
            f"resourcegroups/{config['resource_group']}/"
            f"workspaces/{config['workspace_name']}/"
            f"datastores/{datastore_name}/paths/{path_suffix}")


def get_azure_openai_config() -> Dict[str, str]:
    """
    Get Azure OpenAI configuration from environment variables
    
    Returns:
        Dict[str, str]: Azure OpenAI configuration
    """
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    return {
        'azure_endpoint': endpoint,
        'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    }
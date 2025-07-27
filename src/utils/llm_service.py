from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from azure.identity import ManagedIdentityCredential, DefaultAzureCredential, get_bearer_token_provider
from typing import Optional, Literal, List, Dict, Union
import os

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass


def get_model_libraries():
    """
    Get model libraries with Azure endpoint loaded from environment variables
    """
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    return {
        'gpt-4o': {
            "model_name": "gpt-4o",
            "engine": "gpt-4o",
            "azure_endpoint": azure_endpoint,
            "api_version": "2024-12-01-preview",
            "use_azure_ad": True
        }, 

        'gpt-4.1': {
            "model_name": "gpt-4.1",
            "engine": "gpt-4.1",
            "azure_endpoint": azure_endpoint,
            "api_version": "2024-12-01-preview",
            "use_azure_ad": True
        },

        'o1': {
            "model_name": "o1",
            "engine": "o1",
            "azure_endpoint": azure_endpoint,
            "api_version": "2024-12-01-preview",
            "use_azure_ad": True
        },

        'o4-mini': {
            "model_name": "o4-mini",
            "engine": "o4-mini",
            "azure_endpoint": azure_endpoint,
            "api_version": "2025-04-01-preview",
            "use_azure_ad": True
        }, 

        'o3': {
            "model_name": "o3",
            "engine": "o3",
            "azure_endpoint": azure_endpoint,
            "api_version": "2025-04-01-preview",
            "use_azure_ad": True
        },

        'o3-mini': {
            "model_name": "o3-mini",
            "engine": "o3-mini",
            "azure_endpoint": azure_endpoint,
            "api_version": "2025-01-01-preview",
            "use_azure_ad": True
        }, 

        'text-embedding-ada-002': {
            "model_name": "text-embedding-ada-002",
            "engine": "text-embedding-ada-002",
            "azure_endpoint": azure_endpoint,
            "api_version": "2025-01-01-preview",
            "use_azure_ad": True
        }
    }

# Environment variable names for personal configuration
ENV_VARS = {
    'AZURE_OPENAI_API_KEY': 'AZURE_OPENAI_API_KEY',
    'AZURE_OPENAI_ENDPOINT': 'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_API_VERSION': 'AZURE_OPENAI_API_VERSION',
    'USE_PERSONAL_CONFIG': 'USE_PERSONAL_CONFIG'  # Set to 'true' to use personal config
}


def get_llm(
    model_name: str = 'gpt-4o', 
    auth_method: Literal['use_azure_ad', 'use_key', 'use_mi'] = 'use_azure_ad',
    api_key: Optional[str] = None,
    mi_client_id: Optional[str] = None,
    use_personal_config: Optional[Dict] = None,
    **llm_kwargs
) -> AzureOpenAI:
    """
    Get a LLM instance based on the model name with support for different authentication methods

    Args:
        model_name (str): The name of the model to use
        auth_method (str): Authentication method - 'use_azure_ad', 'use_key', or 'use_mi'
        api_key (str, optional): API key when auth_method is 'use_key'
        mi_client_id (str, optional): Managed Identity client ID when auth_method is 'use_mi'
        use_personal_config (dict, optional): If provided, will override the default model configuration. The format should be the same as the MODEL_LIBRARIES dictionary.
        **llm_kwargs: Additional keyword arguments to pass to the LLM constructor

    Returns:
        AzureOpenAI: A LLM instance
    """
    if use_personal_config is None: 
        model_libraries = get_model_libraries()
        if model_name not in model_libraries:
            raise ValueError(f"Model {model_name} not found in the model library")
        model_config = model_libraries.get(model_name)
    else:
        model_config = use_personal_config.get(model_name)
    if auth_method == 'use_azure_ad':
        # Use Azure AD authentication
        azure_ad_token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        llm = AzureOpenAI(
            model=model_config["model_name"],
            engine=model_config["engine"],
            azure_endpoint=model_config["azure_endpoint"],
            api_version=model_config["api_version"],
            use_azure_ad=model_config["use_azure_ad"],
            azure_ad_token_provider=azure_ad_token_provider,
            **llm_kwargs
        )
    elif auth_method == 'use_mi':
        # use Managed Identity authentication
        if not mi_client_id:
            raise ValueError("Managed Identity client ID is required when using 'use_mi' authentication method.")
        azure_ad_token_provider = get_bearer_token_provider(
            ManagedIdentityCredential(client_id=mi_client_id),
            "https://cognitiveservices.azure.com/.default"
        )
        llm = AzureOpenAI(
            model=model_config["model_name"],
            engine=model_config["engine"],
            azure_endpoint=model_config["azure_endpoint"],
            api_version=model_config["api_version"],
            use_azure_ad=True,  # Managed Identity uses Azure AD
            azure_ad_token_provider=azure_ad_token_provider,
            **llm_kwargs
        )
    else:
        # Use API key authentication
        if not api_key:
            raise ValueError("API key is required when using 'use_key' authentication method.")
        llm = AzureOpenAI(
            model=model_config["model_name"],
            engine=model_config["engine"],
            azure_endpoint=model_config["azure_endpoint"],
            api_version=model_config["api_version"],
            api_key=api_key,
            **llm_kwargs
        )
    return llm
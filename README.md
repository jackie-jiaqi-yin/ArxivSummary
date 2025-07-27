# ArxivSummary

ArxivSummary is a comprehensive Python project that crawls arXiv papers, generates AI-powered summaries, and supports both local execution and Azure ML Pipeline deployment for scalable, automated processing.

## 🚀 Features

- **ArXiv Paper Crawling**: Search and download papers based on custom queries
- **AI-Powered Summarization**: Generate comprehensive summaries using advanced language models
- **HTML Report Generation**: Create formatted HTML reports for easy viewing
- **Dual Deployment Options**:
  - **Local Execution**: Perfect for development and small-scale processing
  - **Azure ML Pipelines**: Enterprise-scale automated processing with scheduling
- **Logic App Integration**: Automated email notifications and workflow triggers
- **Flexible Authentication**: Support for API keys, Azure AD, and Managed Identity

## 📋 Prerequisites

### For Local Usage
- Python 3.10+
- Conda (for environment management)
- Azure OpenAI account with API access

### For Azure ML Pipeline Usage
- Azure subscription
- Azure Machine Learning workspace
- Azure Key Vault (for secrets management)
- Azure Logic Apps (optional, for email notifications)
- User-assigned Managed Identity with appropriate permissions

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jackie-jiaqi-yin/ArxivSummary.git
   cd ArxivSummary
   ```

2. **Create and activate the Conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate ArxivSummary
   ```

3. **Configure your settings** (see Configuration section below)

## ⚙️ Configuration

### Local Configuration

1. **Copy and configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env file with your actual Azure credentials
   ```

2. **Set up authentication** in `config.yml`:
   ```yaml
   summary:
     auth_method: 'use_key'  # Options: 'use_azure_ad', 'use_key', 'use_mi'
     # For 'use_key': Set AZURE_OPENAI_API_KEY in .env file
     # For 'use_mi': Set AZURE_MANAGED_IDENTITY_CLIENT_ID in .env file
     mi_client_id: null  # Will be loaded from .env when auth_method is 'use_mi'
   ```

3. **Required environment variables** in `.env` file:
   ```bash
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   ```

### Azure ML Pipeline Configuration

**Option 1: For Local Testing and Submission (Recommended)**
Use environment variables by adding to your `.env` file:
```bash
# Azure ML Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP_NAME=your-resource-group
AZURE_AML_WORKSPACE_NAME=your-workspace-name
AZURE_DATASTORE_NAME=workspaceblobstore
AZURE_COMPUTE_CLUSTER_NAME=cpu-cluster

# Key Vault Configuration (for Logic Apps)
AZURE_KEYVAULT_NAME=your-keyvault-name
```

**Option 2: Manual Configuration (For Production Deployment)**

1. **Update Azure resource identifiers** in the following files:

   **AMLPipeline/config.json**:
   ```json
   {
       "subscription_id": "YOUR_AZURE_SUBSCRIPTION_ID",
       "resource_group": "YOUR_RESOURCE_GROUP_NAME",
       "workspace_name": "YOUR_AML_WORKSPACE_NAME"
   }
   ```

2. **Update Azure DevOps pipelines** in `azure-agentic-ai-pipelines.yml` and `azure-pipelines.yml`:
   - Replace `YOUR_AZURE_SUBSCRIPTION_ID`
   - Replace `YOUR_AZURE_SERVICE_CONNECTION_NAME`
   - Replace `YOUR_RESOURCE_GROUP_NAME`
   - Replace `YOUR_AML_WORKSPACE_NAME`

## 🏃‍♂️ Usage

### Local Execution

1. **Configure your search parameters** in `config.yml`:
   ```yaml
   arxiv_crawl:
     run: True
     pdf_download: False
     query: '(cat:cs.CL OR cat:cs.AI or cat:cs.LG) AND (ti:"ai agent" OR abs:"ai agent")'
     latest_num_papers: 100
     output_dir: 'data/arxiv'

   summary:
     run: True
     auth_method: 'use_key'
     model_name: 'gpt-4o'
     batch_size: 20
     max_concurrent_batches: 5
     output_dir: 'data/arxiv'
   ```

2. **Run the pipeline**:
   ```bash
   python main.py --config config.yml
   ```

### Azure ML Pipeline Execution

**Prerequisites**: Ensure your `.env` file contains Azure ML configuration variables.

1. **Test individual components**:
   ```bash
   cd AMLPipeline
   python test_submit_arxiv_crawl.py      # Test paper crawling
   python test_submit_summary_pipeline.py # Test summarization
   python test_submit_logic_app_trigger.py # Test notifications
   python test_submit_post_process.py     # Test data archiving
   ```

2. **Submit full pipelines**:
   ```bash
   cd AMLPipeline
   python submit_agentic_ai_pipeline.py   # AI Agent research (bi-monthly)
   python submit_llm_pipeline.py          # LLM research (bi-weekly)
   ```

3. **For automated scheduling**, use the Azure DevOps pipelines:
   - `azure-agentic-ai-pipelines.yml`: Runs 1st and 15th of every month
   - `azure-pipelines.yml`: Runs every Tuesday and Thursday

## 📁 Project Structure

```
ArxivSummary/
├── main.py                              # Local execution entry point
├── config.yml                           # Main configuration file
├── environment.yml                      # Conda environment specification
├── src/
│   ├── arxiv_crawler.py                # ArXiv paper crawling logic
│   ├── summarizer.py                   # AI-powered summarization
│   ├── html_generator.py               # HTML report generation
│   └── utils/
│       ├── config.py                   # Configuration management
│       ├── helpers.py                  # Utility functions
│       └── llm_service.py              # LLM service abstraction
├── AMLPipeline/                        # Azure ML Pipeline components
│   ├── config.json                     # Azure resource configuration
│   ├── submit_agentic_ai_pipeline.py   # Submit AI agent pipeline
│   ├── submit_llm_pipeline.py          # Submit LLM pipeline
│   ├── arxiv_crawl/                    # Crawling component
│   ├── summary/                        # Summarization component
│   ├── post_process/                   # Post-processing component
│   └── logic_app_trigger/              # Logic App integration
├── azure-agentic-ai-pipelines.yml      # Azure DevOps pipeline (bi-monthly)
├── azure-pipelines.yml                 # Azure DevOps pipeline (bi-weekly)
└── data/                               # Output data directory
```

## 📊 Output

The pipeline generates the following outputs:

- **catalog.csv**: Structured data of crawled papers
- **catalog.md**: Markdown-formatted paper catalog
- **abstract_summary.md**: AI-generated comprehensive summary
- **abstract_summary.html**: Formatted HTML report (ready for email/web)

## 🔧 Advanced Configuration

### Authentication Methods

All authentication is now configured via the `.env` file:

1. **API Key** (`use_key`) - Most common for local development:
   - Set `AZURE_OPENAI_API_KEY` in your `.env` file
   - Set `auth_method: 'use_key'` in `config.yml`

2. **Azure AD** (`use_azure_ad`) - For production environments:
   - Uses DefaultAzureCredential for authentication
   - Set `auth_method: 'use_azure_ad'` in `config.yml`
   - Requires appropriate Azure AD permissions

3. **Managed Identity** (`use_mi`) - For Azure ML environments:
   - Set `AZURE_MANAGED_IDENTITY_CLIENT_ID` in your `.env` file
   - Set `auth_method: 'use_mi'` in `config.yml`

### Custom Queries

ArXiv query syntax examples:
```yaml
# AI Agent papers
query: '(cat:cs.CL OR cat:cs.AI) AND (ti:"ai agent" OR abs:"ai agent")'

# Large Language Models
query: '(cat:cs.CL OR cat:cs.AI) AND (ti:"large language model" OR abs:"large language model" OR ti:LLM)'

# Computer Vision + AI
query: '(cat:cs.CV OR cat:cs.AI) AND submittedDate:[20240101 TO 20241231]'
```

## 🔐 Security Configuration

### Required Azure Resources

1. **Azure Key Vault**: Store Logic App webhook URLs and other secrets
2. **Managed Identity**: For secure authentication without storing credentials
3. **Service Principal**: For Azure DevOps pipeline authentication
4. **Azure Logic Apps**: For automated sending the email of HTML reports.
5. **Azure Blob Storage**: For storing crawled papers and generated summaries
### Environment Variables

**Critical Security Note**: Never commit the `.env` file to version control!

Add `.env` to your `.gitignore` file. The following sensitive values should only be in your local `.env` file:
- `AZURE_OPENAI_API_KEY`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP_NAME`
- `AZURE_AML_WORKSPACE_NAME`
- `AZURE_KEYVAULT_NAME`
- `AZURE_MANAGED_IDENTITY_CLIENT_ID`

## 🚀 Deployment

### Azure ML Pipeline Deployment (CI/CD)

1. **Set up Azure resources**:
   - Create Azure ML workspace
   - Configure compute clusters (e.g. `cpu-cluster`)
   - Set up datastores for input/output

2. **Configure permissions**:
   - Grant Managed Identity access to Azure OpenAI
   - Grant access to Key Vault for secrets
   - Configure Logic App permissions

3. **Deploy via Azure DevOps**:
   - Import pipeline YAML files
   - Configure service connections
   - Set up variable groups for configuration

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update documentation as needed
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Authentication failures**: Verify your API keys and managed identity configuration
2. **Pipeline failures**: Check Azure ML compute cluster status and quotas
3. **Logic App not triggering**: Verify Key Vault access and webhook URLs

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Azure ML pipeline logs
3. Open an issue in the repository

---

**Note**: This project requires proper Azure configuration. Ensure all placeholder values (YOUR_*) are replaced with actual resource identifiers before deployment.
import sys
import os
import json
import datetime
import requests
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

# Get the directory three levels up
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append it to sys.path
sys.path.append(parent_dir)

import argparse


def trigger_logic_app(input_dir, 
                      output_dir,
                      logic_app_name,
                      blob_path_prefix):
    """
    Trigger Azure Logic App with pipeline completion notification
    
    Args:
        input_dir: Directory containing pipeline results and HTML file
        output_dir: Directory to write trigger results
    """
    
    # Retrieve Logic App webhook URL from Azure Key Vault using user-assigned managed identity
    # Load Key Vault URL from environment variable, fallback to placeholder
    keyvault_name = os.getenv('AZURE_KEYVAULT_NAME', 'YOUR_KEYVAULT_NAME')
    key_vault_url = f"https://{keyvault_name}.vault.azure.net/"
    
    # Try different managed identity approaches for compute instance vs compute cluster
    try:
        # First try without client_id (works on compute instance)
        credential = ManagedIdentityCredential()
        secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
        # Test the credential by attempting to get the secret
        logic_app_url = secret_client.get_secret(logic_app_name).value
    except Exception:
        # If that fails, try with specific client_id (needed for compute cluster)
        # Load managed identity client ID from environment variable, fallback to placeholder
        mi_client_id = os.getenv('AZURE_MANAGED_IDENTITY_CLIENT_ID', 'YOUR_MANAGED_IDENTITY_CLIENT_ID')
        credential = ManagedIdentityCredential(client_id=mi_client_id)
        secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
        logic_app_url = secret_client.get_secret(logic_app_name).value
    

    
    # Find the HTML file in the input directory
    html_file_path = None
    html_filename = None
    for file in os.listdir(input_dir):
        if file.endswith('.html'):
            html_filename = file
            break
    
    if not html_filename:
        print("Warning: No HTML file found in input directory")
        html_file_path = ""
    else:
        # Send only the blob path relative to the container for Logic App's get_blob_content action
        html_file_path = f"{blob_path_prefix}/{html_filename}"
    
    # Prepare the payload for Logic App
    payload = {
        "pipeline_name": "ArXiv Summary Pipeline",
        "status": "completed",
        "html_file_path": html_file_path,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    print(f"Triggering Logic App with payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make HTTP POST request to Logic App
        response = requests.post(
            logic_app_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            print("Successfully triggered Logic App")
            result = {
                "status": "success",
                "response_code": response.status_code,
                "response_text": response.text,
                "payload": payload,
                "timestamp": datetime.datetime.now().isoformat()
            }
        else:
            print(f"Failed to trigger Logic App. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            result = {
                "status": "error",
                "response_code": response.status_code,
                "response_text": response.text,
                "payload": payload,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Error triggering Logic App: {str(e)}")
        result = {
            "status": "error",
            "error_message": str(e),
            "payload": payload,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    # Write result to output directory
    result_file = os.path.join(output_dir, 'logic_app_trigger_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Trigger result saved to: {result_file}")


def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--logic_app_name', type=str, required=True)
    parser.add_argument('--blob_path_prefix', type=str, required=True)
    return parser.parse_args()


def main(args):
    # Trigger Logic App
    trigger_logic_app(args.input_dir, args.output_dir, args.logic_app_name, args.blob_path_prefix)


if __name__ == '__main__':
    args = parse_parser()
    main(args)
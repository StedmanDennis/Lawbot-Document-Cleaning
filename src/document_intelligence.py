#https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/use-sdk-rest-api?view=doc-intel-4.0.0&tabs=windows&pivots=programming-language-python
import os
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

key_env_var_name: str = 'DOCUMENT_INTELLIGENCE_KEY'
endpoint_env_var_name: str = 'DOCUMENT_INTELLIGENCE_ENDPOINT' 
key = os.getenv(key_env_var_name)
endpoint = os.getenv(endpoint_env_var_name)

def document_intelligence_extract(filePath: Path):
    if key == None:
        raise Exception(f'Document Intelligence Key missing.\nAdd your Azure Document Intelligence Key as an environment variable named {key_env_var_name}')
    if endpoint == None:
        raise Exception(f'Document Intelligence Endpoint missing.\nAdd your Azure Document Intelligence Endpoint as an environment variable named {endpoint_env_var_name}')
    
    client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    fileBinary = filePath.open("rb")

    poller = client.begin_analyze_document(
        "prebuilt-read", document=fileBinary
    )

    result = poller.result()

    return result
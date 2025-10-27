import os
import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.artifacts import InMemoryArtifactService # Or GcsArtifactService
from google.adk import a2a

# Google Cloud Imports
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage

from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This is okay for local testing, but will not be used in deployment.
# The environment variables will be injected into the runtime by Vertex AI.
load_dotenv()

# --- NO GLOBAL CONFIGS OR CLIENTS ---
# All variables will be loaded from the environment at *runtime* inside the functions.

artifact_service = InMemoryArtifactService()

def get_credentials():
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    DATA_STORE_ID = os.getenv("VERTEX_AI_DATA_STORE_ID")

    # Build service account info from env vars *at runtime*
    # Ensure private_key is properly formatted
    private_key = os.getenv("private_key")
    if private_key:
        private_key = private_key.replace('\\n', '\n')

    service_account_info = {
        "project_id": GCP_PROJECT_ID,
        "private_key_id": os.getenv("private_key_id"),
        "private_key": private_key,
        "client_email": os.getenv("client_email"),
        "client_id": os.getenv("client_id"),
        "auth_uri": os.getenv("auth_uri"),
        "token_uri": os.getenv("token_uri"),
        "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
        "client_x509_cert_url": os.getenv("client_x509_cert_url"),
        "universe_domain": os.getenv("universe_domain")
    }
    
    # Check if all required keys for service account are present
    if not all([GCP_PROJECT_ID, DATA_STORE_ID, private_key, os.getenv("client_email")]):
        logger.error("Missing service account details or config in environment variables.")
        return {"citations": [], "error": "Server configuration error."}

    return service_account.Credentials.from_service_account_info(service_account_info) 

def read_gcs_file(gcs_uri: str) -> bytes:
    """
    Reads a file from Google Cloud Storage and returns its content as bytes.
    """
    # Initialize client inside the function
    storage_client = storage.Client(credentials=get_credentials())
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes(),blob_name

def extract_and_ingest_contract(gcs_uri: str) -> bool:
    """
    Extracts content from a PDF, uploads the text to a GCS bucket,
    creates a sanitized metadata file, and ingests into Vertex AI Search from the bucket.
    """
    try:
        # --- 1. Load Configuration at RUNTIME ---
        GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        GCP_LOCATION = os.getenv("GCP_LOCATION")
        DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID")
        DATA_STORE_ID = os.getenv("VERTEX_AI_DATA_STORE_ID")
        BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

        if not all([GCP_PROJECT_ID, GCP_LOCATION, DOCAI_PROCESSOR_ID, DATA_STORE_ID, BUCKET_NAME]):
            logger.error("Missing one or more required environment variables for ingestion.")
            return False

        # --- 2. Document AI: Extract Text ---
        logger.info(f"Starting document extraction for: {gcs_uri}")
        docai_client = documentai.DocumentProcessorServiceClient(credentials=get_credentials(),
            client_options=ClientOptions(api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com")
        )
        processor_name = docai_client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, DOCAI_PROCESSOR_ID)
        image_content,original_filename = read_gcs_file(gcs_uri)
        raw_document = documentai.RawDocument(content=image_content, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = docai_client.process_document(request=request)
        extracted_text = result.document.text
        logger.info("Successfully extracted text from document.")


        # --- 3. GCS: Prepare, Sanitize, and Upload Metadata for Ingestion ---
        logger.info(f"Preparing and uploading ingestion metadata to GCS bucket: {BUCKET_NAME}")
        storage_client = storage.Client(credentials=get_credentials())
        bucket = storage_client.bucket(BUCKET_NAME)
        
        sanitized_id = re.sub(r'[^a-zA-Z0-9-_]', '_', original_filename)
        logger.info(f"Original filename '{original_filename}' sanitized to ID '{sanitized_id}'")
        
        metadata_filename = f"{sanitized_id}.txt"
        
        blob = bucket.blob(metadata_filename)
        blob.upload_from_string(extracted_text) 
        gcs_uri_for_ingestion = f"gs://{BUCKET_NAME}/{metadata_filename}"
        logger.info(f"Successfully uploaded metadata for ingestion to {gcs_uri_for_ingestion}")


        # --- 4. Vertex AI Search: Ingest from GCS Bucket ---
        logger.info(f"Starting ingestion into data store: {DATA_STORE_ID} from GCS")
        discovery_client = discoveryengine.DocumentServiceClient(credentials=get_credentials())
        parent = discovery_client.branch_path(
            project=GCP_PROJECT_ID,
            location="global", # Ingestion is typically 'global' location
            data_store=DATA_STORE_ID,
            branch="default_branch",
        )
        
        gcs_source = discoveryengine.GcsSource(input_uris=[gcs_uri_for_ingestion], data_schema="content")

        ingest_request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            gcs_source=gcs_source,
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )
        res=discovery_client.import_documents(request=ingest_request)
        logger.info(f"Ingestion operation started: {res.operation.name}")
        # print(res.result()) # Avoid blocking in an agent, just log the start
        logger.info(f"Successfully started ingestion from GCS for document ID: {sanitized_id}")

        return True

    except Exception as e:
        logger.error(f"An error occurred during document processing: {e}", exc_info=True)
        return False


# --- Real Vertex AI Search Tool ---
class VertexAiSearchTool(BaseTool):
    """A tool to search for citations in an ingested document using Vertex AI Search."""
    def __init__(self):
        super().__init__(
            name="vertex_ai_search",
            description="Searches for relevant citations in the contract document to support the requirement and test case question."
        )
        # DO NOT initialize clients or credentials here.


    async def _arun(self, requirement: str, test_case_question: str) -> Dict[str, Any]:
        """
        Performs a search against a Vertex AI Search data store and returns citations.
        """
        logger.info(f"Searching with Requirement: '{requirement}' and Question: '{test_case_question}'")
        
        try:
            # --- 1. Load Configuration at RUNTIME ---
            GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
            DATA_STORE_ID = os.getenv("VERTEX_AI_DATA_STORE_ID")

            # Build service account info from env vars *at runtime*
            # Ensure private_key is properly formatted
            private_key = os.getenv("private_key")
            if private_key:
                private_key = private_key.replace('\\n', '\n')

            service_account_info = {
                "project_id": GCP_PROJECT_ID,
                "private_key_id": os.getenv("private_key_id"),
                "private_key": private_key,
                "client_email": os.getenv("client_email"),
                "client_id": os.getenv("client_id"),
                "auth_uri": os.getenv("auth_uri"),
                "token_uri": os.getenv("token_uri"),
                "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
                "client_x509_cert_url": os.getenv("client_x509_cert_url"),
                "universe_domain": os.getenv("universe_domain")
            }
            
            # Check if all required keys for service account are present
            if not all([GCP_PROJECT_ID, DATA_STORE_ID, private_key, os.getenv("client_email")]):
                logger.error("Missing service account details or config in environment variables.")
                return {"citations": [], "error": "Server configuration error."}

            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
            # --- 2. Initialize Clients at RUNTIME ---
            search_client = discoveryengine.SearchServiceClient(credentials=credentials)
            serving_config = search_client.serving_config_path(
                project=GCP_PROJECT_ID,
                location="global", # Serving configs are typically 'global'
                data_store=DATA_STORE_ID,
                serving_config="default_config",
            )
            
            # Combine requirement and question for a more effective query
            query = f"{requirement} {test_case_question}"

            search_request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=5, # Get top 5 relevant chunks
                content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                    snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=True
                    ),
                    extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                        max_extractive_answer_count=3
                    )
                )
            )

            response = search_client.search(request=search_request)
            print(response) # For debugging in Vertex AI logs
            
            citations = []
            for i, result in enumerate(response.results):
                # Check if snippets exist
                if 'snippets' in result.document.derived_struct_data and result.document.derived_struct_data['snippets']:
                    snippet = result.document.derived_struct_data['snippets'][0]['snippet']
                    citations.append({
                        "citation_id": f"cite_{i+1}",
                        "citation_text": snippet.strip().replace('\n', ' ')
                    })

            logger.info(f"Found {len(citations)} citations.")
            return {"citations": citations}

        except Exception as e:
            logger.error(f"An error occurred during Vertex AI search: {e}", exc_info=True)
            return {"citations": [], "error": str(e)}

    async def query(self, requirement: str, test_case_question: str) -> Dict[str, Any]:
        return await self._arun(requirement, test_case_question)


# --- This instantiation is now SAFE ---
vertex_ai = VertexAiSearchTool()


vertex_ai_search=FunctionTool(vertex_ai.query)

instruction = """
1. You must first ask the user for the 'Requirement' and the 'Test Case Question'. Do not proceed until you have both.

2. After receiving the 'Requirement' and 'Test Case Question':
    - If the user also provides a 'contract document GCS URI', first perform `extract_and_ingest_contract` using the provided URI.
    - If no GCS URI is provided, skip ingestion and proceed directly.

3. Next, use the `vertex_ai_search` tool with the given 'Requirement' and 'Test Case Question' to find relevant citations.

4. Analyze the citations returned by `vertex_ai_search`.

5. Generate the final output in the following JSON format:
   {
     "summary": "A concise summary of how the citations support the requirement.",
     "citations": [
       {"citation_id": "<id>", "citation_text": "<text>","meta_data":{<metadata>}},
       ...
     ]
   }

   Note : citations are direct results from `vertex_ai_search`
            meta_data is all other remaing data from result of tool call
"""



# --- Agent Definition (This is now SAFE to define) ---
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name='RiskAnalysisAgent',
    description='Analyzes contract documents to find supporting citations for a given requirement and test case question.',
    instruction=instruction,
    tools=[vertex_ai_search,extract_and_ingest_contract],
)


import os
import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.artifacts import InMemoryArtifactService # Or GcsArtifactService


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

#load environment variables from .env file
load_dotenv()

# --- Load Configuration from Environment ---
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID")
DATA_STORE_ID = os.getenv("VERTEX_AI_DATA_STORE_ID")

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

artifact_service = InMemoryArtifactService()
 

def read_gcs_file(gcs_uri: str) -> bytes:
    """
    Reads a file from Google Cloud Storage and returns its content as bytes.
    """
    storage_client = storage.Client()
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
        # 1. --- Document AI: Extract Text (Unchanged) ---
        logger.info(f"Starting document extraction for: {gcs_source}")
        # ... (all the documentai client code remains the same) ...
        docai_client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com")
        )
        processor_name = docai_client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, DOCAI_PROCESSOR_ID)
        image_content,original_filename = read_gcs_file(gcs_uri)
        raw_document = documentai.RawDocument(content=image_content, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = docai_client.process_document(request=request)
        extracted_text = result.document.text
        logger.info("Successfully extracted text from document.")


        # 2. --- GCS: Prepare, Sanitize, and Upload Metadata for Ingestion ---
        logger.info(f"Preparing and uploading ingestion metadata to GCS bucket: {BUCKET_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        sanitized_id = re.sub(r'[^a-zA-Z0-9-_]', '_', original_filename)
        logger.info(f"Original filename '{original_filename}' sanitized to ID '{sanitized_id}'")
        
        metadata_filename = f"{sanitized_id}.txt"
        
        # Upload the JSONL content to a file in GCS. json.dumps() is only called once.
        blob = bucket.blob(metadata_filename)
        blob.upload_from_string(extracted_text) # This creates one line with a valid JSON object
        gcs_uri_for_ingestion = f"gs://{BUCKET_NAME}/{metadata_filename}"
        logger.info(f"Successfully uploaded metadata for ingestion to {gcs_uri_for_ingestion}")


        # 3. --- Vertex AI Search: Ingest from GCS Bucket (Unchanged) ---
        logger.info(f"Starting ingestion into data store: {DATA_STORE_ID} from GCS")
        # ... (the rest of the function is the same) ...
        discovery_client = discoveryengine.DocumentServiceClient()
        parent = discovery_client.branch_path(
            project=GCP_PROJECT_ID,
            location="global",
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
        print(res.result())
        logger.info(f"Successfully started ingestion from GCS for document ID: {sanitized_id}")

        return True

    except Exception as e:
        logger.error(f"An error occurred during document processing: {e}")
        return False


# --- Real Vertex AI Search Tool ---
class VertexAiSearchTool(BaseTool):
    """A tool to search for citations in an ingested document using Vertex AI Search."""
    def __init__(self):
        super().__init__(
            name="vertex_ai_search",
            description="Searches for relevant citations in the contract document to support the requirement and test case question."
        )
        self.credentials=service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)


    async def _arun(self, requirement: str, test_case_question: str) -> Dict[str, Any]:
        """
        Performs a search against a Vertex AI Search data store and returns citations.
        """
        logger.info(f"Searching with Requirement: '{requirement}' and Question: '{test_case_question}'")
        
        try:
            search_client = discoveryengine.SearchServiceClient(credentials=self.credentials)
            serving_config = search_client.serving_config_path(
                project=GCP_PROJECT_ID,
                location="global",
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
            print(response)
            
            citations = []
            for i, result in enumerate(response.results):
                snippet = result.document.derived_struct_data['snippets'][0]['snippet']
                citations.append({
                    "citation_id": f"cite_{i+1}",
                    "citation_text": snippet.strip().replace('\n', ' ')
                })

            logger.info(f"Found {len(citations)} citations.")
            return {"citations": citations}

        except Exception as e:
            logger.error(f"An error occurred during Vertex AI search: {e}")
            return {"citations": [], "error": str(e)}
    async def query(self, requirement: str, test_case_question: str) -> Dict[str, Any]:
        return await self._arun(requirement, test_case_question)


vertex_ai_search = VertexAiSearchTool()


instructions="""
    1 .You must first ask the user for the 'Requirement' and the 'Test Case Question'. Do not proceed until you have both.
    2 .Now, ask the user to provide the contract document gcs uri for analysis
    3.The document has been ingested. Your task is to use the `vertex_ai_search` tool to find citations.

        1. **Call the Tool**: Use the `vertex_ai_search` tool with the provided 'Requirement' and 'Test Case Question'.

        2. **Analyze Citations**: Review the citations returned by the tool.

        3. **Generate Output**: Create a final response in the following JSON format:
           {
             "summary": "A concise summary of how the citations support the requirement.",
             "citations": [
               {"citation_id": "<id>", "citation_text": "<text>"},
               ...
             ]
           }
"""

# # --- Pydantic Model for Agent Response ---
class AgentResponse(BaseModel):
    summary: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None

# --- Agent Definition ---
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name='RiskAnalysisAgent',
    description='Analyzes contract documents to find supporting citations for a given requirement and test case question.',
    instruction=instructions,
    tools=[vertex_ai_search.query,extract_and_ingest_contract],
)



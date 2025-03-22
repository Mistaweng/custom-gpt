import os
from dotenv import load_dotenv
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import tiktoken

# Load environment variables
load_dotenv()

# Initialize Pinecone with new API
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

# Constants
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
MODEL_NAME = "text-embedding-ada-002"
EMBEDDING_COST_PER_1000_TOKENS = 0.0004 / 1000
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Embedding setup
embedding = OpenAIEmbeddings(model=MODEL_NAME)

# Data directory
current_dir = os.path.dirname(__file__)
docs_dir = os.path.join(current_dir, "data")

def calculate_embedding_cost(chunks):
    """Calculate the cost of embeddings for the given chunks."""
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    total_tokens = sum(len(encoding.encode(chunk.page_content)) for chunk in chunks)
    cost = total_tokens * EMBEDDING_COST_PER_1000_TOKENS
    return round(cost, 7)

def ensure_index_exists():
    """Ensure the Pinecone index exists, create it if it doesn't."""
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine'
        )
    return pc.Index(INDEX_NAME)

def process_documents(dry_run=False):
    """Estimate cost and optionally ingest data into Pinecone."""
    total_estimated_cost = 0
    
    # Ensure index exists if not dry run
    if not dry_run:
        ensure_index_exists()
    
    for root, _, files in os.walk(docs_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                try:
                    logger.info(f"Processing file: {filename}")
                    loader = PyPDFLoader(file_path=file_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    chunks = text_splitter.split_documents(data)
                    
                    # Calculate embedding cost
                    cost = calculate_embedding_cost(chunks)
                    total_estimated_cost += cost
                    logger.info(f"Cost for {filename}: ${cost}")
                    
                    # Ingest into Pinecone if not a dry run
                    if not dry_run:
                        Pinecone.from_documents(
                            documents=chunks,
                            embedding=embedding,
                            index_name=INDEX_NAME
                        )
                        logger.info(f"Ingested {filename} into Pinecone.")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    
    return total_estimated_cost

if __name__ == "__main__":
    logger.info("Estimating embedding cost...")
    total_cost = process_documents(dry_run=True)
    logger.info(f"Total estimated cost: ${total_cost}")
    
    user_input = input("Would you like to continue with ingestion? (y/n): ").strip().lower()
    if user_input == "y":
        logger.info("Starting ingestion into Pinecone...")
        process_documents(dry_run=False)
        logger.info("Ingestion completed successfully.")
    else:
        logger.info("Operation aborted by user.")
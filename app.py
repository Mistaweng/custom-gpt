import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Setup environment variables and clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone with new API
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))

# Middleware to secure HTTP endpoint
security = HTTPBearer()

def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

class QueryModel(BaseModel):
    query: str

@app.post("/")
async def get_context(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        # Convert query to embeddings
        res = openai_client.embeddings.create(
            input=[query_data.query], 
            model="text-embedding-ada-002"
        )
        embedding = res.data[0].embedding

        # Search for matching Vectors
        results = index.query(
            vector=embedding,
            top_k=6,
            include_metadata=True
        )

        # Filter out metadata from search result
        context = [match["metadata"]["text"] for match in results["matches"]]

        return context

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

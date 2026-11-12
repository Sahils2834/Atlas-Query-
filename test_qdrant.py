
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Qdrant environment variables are missing.")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

try:
    print("Connecting to Qdrant...")
    collections = client.get_collections()

    print("Connection successful!")
    print("Existing collections:")

    for collection in collections.collections:
        print("-", collection.name)

except Exception as error:
    print("Qdrant connection failed:", error)


import os
import re
import uuid
import hashlib
from pathlib import Path
from collections import Counter

import fitz
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "rag_documents_v2"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

CHUNK_SIZE_WORDS = int(
    os.getenv("CHUNK_SIZE_WORDS", "350")
)

CHUNK_OVERLAP_WORDS = int(
    os.getenv("CHUNK_OVERLAP_WORDS", "60")
)

BATCH_SIZE = 64
VECTOR_SIZE = 384

SKIP_REFERENCES = True

# --------------------------------------------------
# QDRANT CONNECTION
# --------------------------------------------------

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be configured."
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=60
    )


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\u00ad", "")
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    return text.strip()


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_pages(pdf_path):
    extracted_pages = []

    try:
        document = fitz.open(pdf_path)

        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))

            if len(text.split()) < 10:
                continue

            extracted_pages.append({
                "page": page_number,
                "text": text
            })

        document.close()

    except Exception as error:
        print(f"Error reading {pdf_path.name}: {error}")

    return extracted_pages


# --------------------------------------------------
# REMOVE REFERENCES SECTION
# --------------------------------------------------

def remove_references(text):
    if not SKIP_REFERENCES:
        return text

    patterns = [
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bworks cited\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match and match.start() > len(text) * 0.55:
            return text[:match.start()].strip()

    return text


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def create_chunks(text):
    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):
        end = min(
            start + CHUNK_SIZE_WORDS,
            len(words)
        )

        chunk = " ".join(words[start:end]).strip()

        if len(chunk.split()) >= 35:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP_WORDS

    return chunks


# --------------------------------------------------
# STABLE POINT IDS
# --------------------------------------------------

def generate_point_id(source, page, chunk_index, text):
    content = (
        f"{source}|{page}|{chunk_index}|{text}"
    )

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, content)
    )


# --------------------------------------------------
# COLLECTION SETUP
# --------------------------------------------------

def ensure_collection(client):
    collections = client.get_collections().collections

    exists = any(
        collection.name == COLLECTION_NAME
        for collection in collections
    )

    if exists:
        info = client.get_collection(COLLECTION_NAME)

        if info.config.params.vectors.size != VECTOR_SIZE:
            raise ValueError(
                f"Collection {COLLECTION_NAME} has "
                f"{info.config.params.vectors.size} dimensions. "
                f"Expected {VECTOR_SIZE}. Create a new collection "
                "with the correct dimensions."
            )

        print(f"Using existing collection: {COLLECTION_NAME}")

    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print(f"Created collection: {COLLECTION_NAME}")

    # Index fields used for metadata filtering.
    for field_name, field_schema in [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("page", models.PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema
            )
        except Exception:
            pass


# --------------------------------------------------
# MAIN INGESTION
# --------------------------------------------------

def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    print("\nConnecting to Qdrant...")
    client = get_qdrant_client()

    ensure_collection(client)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    all_records = []

    total_pages = 0

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)

        total_pages += len(pages)

        for page_data in pages:
            page_text = remove_references(
                page_data["text"]
            )

            chunks = create_chunks(page_text)

            for chunk_index, chunk in enumerate(chunks):
                point_id = generate_point_id(
                    pdf_path.name,
                    page_data["page"],
                    chunk_index,
                    chunk
                )

                all_records.append({
                    "id": point_id,
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "chunk_index": chunk_index
                })

    if not all_records:
        print("No valid text chunks were extracted.")
        return

    print(f"\nTotal pages processed: {total_pages}")
    print(f"Total chunks created: {len(all_records)}")

    print("\nGenerating embeddings and uploading to Qdrant...")

    for start in tqdm(
        range(0, len(all_records), BATCH_SIZE)
    ):
        batch = all_records[
            start:start + BATCH_SIZE
        ]

        texts = [
            item["text"] for item in batch
        ]

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        points = []

        for item, embedding in zip(batch, embeddings):
            points.append(
                models.PointStruct(
                    id=item["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": item["text"],
                        "source": item["source"],
                        "page": item["page"],
                        "chunk_index": item["chunk_index"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    info = client.get_collection(COLLECTION_NAME)

    print("\n====================================")
    print("INGESTION COMPLETED")
    print("====================================")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages: {total_pages}")
    print(f"Chunks created: {len(all_records)}")
    print(f"Qdrant points: {info.points_count}")
    print("====================================")


if __name__ == "__main__":
    main()

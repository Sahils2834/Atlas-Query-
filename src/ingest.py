
import os
import re
import uuid
from pathlib import Path

import fitz
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


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

# Smaller batches reduce peak memory usage.
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "16"))

VECTOR_SIZE = 384

SKIP_REFERENCES = True

MIN_CHUNK_WORDS = 35


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

    replacements = {
        "\u00ad": "",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "–": "-",
        "—": "-"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Join words split across PDF lines.
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_pages(pdf_path):

    try:
        with fitz.open(pdf_path) as document:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = clean_text(
                    page.get_text("text")
                )

                if len(text.split()) < 10:
                    continue

                yield {
                    "page": page_number,
                    "text": text
                }

    except Exception as error:

        print(
            f"Error reading {pdf_path.name}: {error}"
        )


# --------------------------------------------------
# REMOVE REFERENCES
# --------------------------------------------------

def remove_references(text):

    if not SKIP_REFERENCES:
        return text

    patterns = [
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bworks cited\b"
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

        chunk = " ".join(
            words[start:end]
        ).strip()

        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP_WORDS

    return chunks


# --------------------------------------------------
# STABLE POINT IDS
# --------------------------------------------------

def generate_point_id(
    source,
    page,
    chunk_index,
    text
):

    content = (
        f"{source}|{page}|{chunk_index}|{text}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            content
        )
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

        info = client.get_collection(
            COLLECTION_NAME
        )

        vector_config = info.config.params.vectors

        if not isinstance(
            vector_config,
            models.VectorParams
        ):
            raise ValueError(
                "The existing collection does not use "
                "a single unnamed dense vector."
            )

        if vector_config.size != VECTOR_SIZE:

            raise ValueError(
                f"Collection {COLLECTION_NAME} has "
                f"{vector_config.size} dimensions. "
                f"Expected {VECTOR_SIZE}."
            )

        print(
            f"Using existing collection: {COLLECTION_NAME}"
        )

    else:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    # Create indexes for metadata filtering.
    for field_name, field_schema in [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("page", models.PayloadSchemaType.INTEGER)
    ]:

        try:

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema
            )

        except Exception as error:

            print(
                f"Index note for {field_name}: {error}"
            )


# --------------------------------------------------
# BUILD RECORDS
# --------------------------------------------------

def build_records(pdf_path):

    for page_data in extract_pdf_pages(pdf_path):

        page_text = remove_references(
            page_data["text"]
        )

        chunks = create_chunks(page_text)

        for chunk_index, chunk in enumerate(chunks):

            yield {
                "id": generate_point_id(
                    pdf_path.name,
                    page_data["page"],
                    chunk_index,
                    chunk
                ),
                "text": chunk,
                "source": pdf_path.name,
                "page": page_data["page"],
                "chunk_index": chunk_index
            }


# --------------------------------------------------
# EMBED AND UPLOAD BATCH
# --------------------------------------------------

def upload_batch(
    client,
    embedding_model,
    records
):

    if not records:
        return 0

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    points = []

    for record, embedding in zip(
        records,
        embeddings
    ):

        points.append(
            models.PointStruct(
                id=record["id"],
                vector=embedding.tolist(),
                payload={
                    "text": record["text"],
                    "source": record["source"],
                    "page": record["page"],
                    "chunk_index": record["chunk_index"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return len(points)


# --------------------------------------------------
# MAIN INGESTION
# --------------------------------------------------

def main():

    pdf_files = sorted(
        DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            f"No PDF files found in {DATA_DIR}"
        )

        return

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    print("\nConnecting to Qdrant...")

    client = get_qdrant_client()

    ensure_collection(client)

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    total_pages = 0
    total_chunks = 0
    uploaded_points = 0

    batch = []

    print("\nStarting ingestion...")

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: {pdf_path.name}"
        )

        for page_data in extract_pdf_pages(pdf_path):

            total_pages += 1

            page_text = remove_references(
                page_data["text"]
            )

            chunks = create_chunks(page_text)

            for chunk_index, chunk in enumerate(chunks):

                record = {
                    "id": generate_point_id(
                        pdf_path.name,
                        page_data["page"],
                        chunk_index,
                        chunk
                    ),
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "chunk_index": chunk_index
                }

                batch.append(record)
                total_chunks += 1

                if len(batch) >= BATCH_SIZE:

                    uploaded_points += upload_batch(
                        client,
                        embedding_model,
                        batch
                    )

                    batch.clear()

                    print(
                        f"Uploaded {uploaded_points} points..."
                    )

    # Upload remaining records.
    if batch:

        uploaded_points += upload_batch(
            client,
            embedding_model,
            batch
        )

        batch.clear()

    info = client.get_collection(
        COLLECTION_NAME
    )

    print("\n====================================")
    print("INGESTION COMPLETED")
    print("====================================")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages processed: {total_pages}")
    print(f"Chunks created: {total_chunks}")
    print(f"Points uploaded: {uploaded_points}")
    print(f"Total Qdrant points: {info.points_count}")
    print("====================================")


if __name__ == "__main__":
    main()

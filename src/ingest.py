
import os
import re
import uuid
import logging

from pathlib import Path
from functools import lru_cache

import fitz

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.embedding import get_embedder


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "atlasquery_documents"
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

# Smaller batches reduce peak memory usage on Render Free.
BATCH_SIZE = max(
    1,
    int(os.getenv("INGEST_BATCH_SIZE", "4"))
)

VECTOR_SIZE = 384

SKIP_REFERENCES = True

MIN_CHUNK_WORDS = 35


# ==================================================
# SHARED FASTEMBED MODEL
# ==================================================

def get_embedding_model():
    """
    Return the shared FastEmbed model.

    The model is initialized and cached in src.embedding.
    Both ingestion and retrieval reuse the same instance.
    """
    return get_embedder()


# ==================================================
# QDRANT CONNECTION
# ==================================================

@lru_cache(maxsize=1)
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


# ==================================================
# TEXT CLEANING
# ==================================================

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


# ==================================================
# PDF EXTRACTION
# ==================================================

def extract_pdf_pages(pdf_path):

    try:
        with fitz.open(pdf_path) as document:
            yield from extract_document_pages(document)

    except Exception:
        logger.exception(
            "Error reading PDF: %s",
            pdf_path
        )
        raise


def extract_pdf_bytes(pdf_bytes):

    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            yield from extract_document_pages(document)

    except Exception:
        logger.exception(
            "Error extracting uploaded PDF."
        )
        raise


def extract_document_pages(document):

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


# ==================================================
# REMOVE REFERENCES
# ==================================================

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


# ==================================================
# CHUNKING
# ==================================================

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

        start = max(
            0,
            end - CHUNK_OVERLAP_WORDS
        )

    return chunks


# ==================================================
# POINT IDS
# ==================================================

def generate_point_id(
    document_id,
    page,
    chunk_index,
    text
):

    content = (
        f"{document_id}|{page}|{chunk_index}|{text}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            content
        )
    )


# ==================================================
# COLLECTION SETUP
# ==================================================

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

        logger.info(
            "Using existing collection: %s",
            COLLECTION_NAME
        )

    else:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        logger.info(
            "Created collection: %s",
            COLLECTION_NAME
        )

    # Index metadata fields used by retrieval filters.
    indexes = [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("page", models.PayloadSchemaType.INTEGER),
        ("document_id", models.PayloadSchemaType.KEYWORD)
    ]

    for field_name, field_schema in indexes:

        try:

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
                wait=True
            )

        except Exception as error:

            logger.info(
                "Payload index note for %s: %s",
                field_name,
                error
            )


# ==================================================
# BUILD RECORDS
# ==================================================

def build_records_from_pages(
    pages,
    filename,
    document_id
):

    for page_data in pages:

        page_text = remove_references(
            page_data["text"]
        )

        chunks = create_chunks(page_text)

        for chunk_index, chunk in enumerate(chunks):

            yield {
                "id": generate_point_id(
                    document_id,
                    page_data["page"],
                    chunk_index,
                    chunk
                ),
                "text": chunk,
                "source": filename,
                "page": page_data["page"],
                "chunk_index": chunk_index,
                "document_id": document_id
            }


def build_records(pdf_path, document_id=None):

    pdf_path = Path(pdf_path)

    if document_id is None:

        document_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"atlasquery-local:{pdf_path.name}"
            )
        )

    yield from build_records_from_pages(
        extract_pdf_pages(pdf_path),
        pdf_path.name,
        document_id
    )


# ==================================================
# EMBED AND UPLOAD BATCH
# ==================================================

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

    # Generate embeddings in a small batch.
    embeddings = list(
        embedding_model.embed(
            texts,
            batch_size=BATCH_SIZE
        )
    )

    if len(embeddings) != len(records):

        raise RuntimeError(
            "Embedding count does not match record count."
        )

    points = []

    for record, embedding in zip(
        records,
        embeddings
    ):

        vector = embedding.tolist()

        if len(vector) != VECTOR_SIZE:

            raise RuntimeError(
                f"Expected {VECTOR_SIZE} dimensions, "
                f"received {len(vector)}."
            )

        points.append(
            models.PointStruct(
                id=record["id"],
                vector=vector,
                payload={
                    "text": record["text"],
                    "source": record["source"],
                    "page": record["page"],
                    "chunk_index": record["chunk_index"],
                    "document_id": record["document_id"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return len(points)


# ==================================================
# INGESTION ENGINE
# ==================================================

def ingest_pages(
    pages,
    filename,
    document_id,
    client=None,
    embedding_model=None
):

    client = client or get_qdrant_client()

    embedding_model = (
        embedding_model or get_embedding_model()
    )

    ensure_collection(client)

    batch = []

    total_chunks = 0
    uploaded_points = 0
    total_pages = 0

    # Process one page at a time.
    # Do not materialize the entire PDF in memory.

    for page_data in pages:

        total_pages += 1

        page_text = remove_references(
            page_data["text"]
        )

        chunks = create_chunks(page_text)

        for chunk_index, chunk in enumerate(chunks):

            record = {
                "id": generate_point_id(
                    document_id,
                    page_data["page"],
                    chunk_index,
                    chunk
                ),
                "text": chunk,
                "source": filename,
                "page": page_data["page"],
                "chunk_index": chunk_index,
                "document_id": document_id
            }

            batch.append(record)

            total_chunks += 1

            if len(batch) >= BATCH_SIZE:

                uploaded_points += upload_batch(
                    client,
                    embedding_model,
                    batch
                )

                # Release batch references immediately.
                batch.clear()

        del chunks
        del page_text

    # Upload remaining records.
    if batch:

        uploaded_points += upload_batch(
            client,
            embedding_model,
            batch
        )

        batch.clear()

    if uploaded_points == 0:

        raise ValueError(
            "No usable text chunks were extracted from this PDF. "
            "The PDF may be scanned or image-only."
        )

    logger.info(
        "Ingested %s: %s pages, %s chunks",
        filename,
        total_pages,
        uploaded_points
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": total_pages,
        "chunks": uploaded_points,
        "status": "completed"
    }


# ==================================================
# PUBLIC UPLOAD FUNCTION
# ==================================================

def ingest_uploaded_pdf(
    pdf_bytes,
    filename,
    document_id=None
):

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if document_id is None:
        document_id = str(uuid.uuid4())

    try:

        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            if document.is_encrypted:

                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            if document.page_count == 0:

                raise ValueError(
                    "The uploaded PDF contains no pages."
                )

            # Stream pages directly into ingestion.
            # Do not convert this generator to list().
            result = ingest_pages(
                pages=extract_document_pages(document),
                filename=filename,
                document_id=document_id
            )

            return result

    except ValueError:
        raise

    except Exception as error:

        logger.exception(
            "Failed to ingest PDF: %s",
            filename
        )

        raise ValueError(
            "Unable to read or process this PDF."
        ) from error


# ==================================================
# LOCAL DATASET INGESTION
# ==================================================

def main():

    pdf_files = sorted(
        DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:

        logger.info(
            "No PDF files found in %s",
            DATA_DIR
        )

        return

    logger.info(
        "Found %s PDF files.",
        len(pdf_files)
    )

    client = get_qdrant_client()

    ensure_collection(client)

    embedding_model = get_embedding_model()

    total_pages = 0
    total_chunks = 0
    uploaded_points = 0

    for pdf_path in pdf_files:

        logger.info(
            "Processing: %s",
            pdf_path.name
        )

        document_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"atlasquery-local:{pdf_path.name}"
            )
        )

        result = ingest_pages(
            pages=extract_pdf_pages(pdf_path),
            filename=pdf_path.name,
            document_id=document_id,
            client=client,
            embedding_model=embedding_model
        )

        total_pages += result["pages"]
        total_chunks += result["chunks"]
        uploaded_points += result["chunks"]

    logger.info(
        "Ingestion completed. "
        "Pages: %s | Chunks: %s | Uploaded: %s",
        total_pages,
        total_chunks,
        uploaded_points
    )


if __name__ == "__main__":
    main()

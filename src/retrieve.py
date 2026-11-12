
import os
import re
import logging
from functools import lru_cache
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

# ==================================================
# CONFIGURATION
# ==================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "rag_documents_v2"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_SIZE = 384

DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
DEFAULT_CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

RRF_K = 60

FILTER_REFERENCE_CHUNKS = True

# ==================================================
# QDRANT CONNECTION
# ==================================================

@lru_cache(maxsize=1)
def get_qdrant_client():

    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is missing from environment variables.")

    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY is missing from environment variables.")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

    return client


def check_collection():

    client = get_qdrant_client()

    info = client.get_collection(COLLECTION_NAME)

    return {
        "points_count": info.points_count or 0,
        "status": str(info.status)
    }


# ==================================================
# MODELS
# ==================================================

@lru_cache(maxsize=1)
def get_embedder():

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker():

    logger.info("Loading CrossEncoder reranker.")

    return CrossEncoder(RERANKER_MODEL)


# ==================================================
# TEXT UTILITIES
# ==================================================

def tokenize(text: str) -> list[str]:

    return re.findall(r"\b\w+\b", text.lower())


def normalize_text(text: str) -> str:

    return " ".join(text.lower().split())


# ==================================================
# REFERENCE FILTERING
# ==================================================

REFERENCE_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)?\.?\s*)?"
    r"(?:references|bibliography|works cited)\s*$",
    re.IGNORECASE
)

REFERENCE_LINE_PATTERNS = [
    re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"\bet al\.\s*\(?\d{4}\)?", re.IGNORECASE),
]


def is_reference_like(candidate: dict) -> bool:

    metadata = candidate.get("metadata") or {}

    section = str(metadata.get("section", "")).lower().strip()

    heading = str(
        metadata.get("section_heading", "")
    ).lower().strip()

    if section in {
        "references",
        "bibliography",
        "works_cited"
    }:
        return True

    if heading in {
        "references",
        "bibliography",
        "works cited"
    }:
        return True

    text = candidate.get("text", "")

    if not text:
        return False

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return False

    if REFERENCE_SECTION_PATTERN.match(lines[0]):
        return True

    reference_lines = 0

    for line in lines:

        has_year = bool(
            re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line)
        )

        has_author = bool(
            re.search(
                r"\b[A-Z][a-z]+,\s*(?:[A-Z]\.?|[A-Z][a-z]+)",
                line
            )
        )

        has_marker = any(
            pattern.search(line)
            for pattern in REFERENCE_LINE_PATTERNS
        )

        if has_marker or (has_year and has_author):
            reference_lines += 1

    ratio = reference_lines / max(len(lines), 1)

    return reference_lines >= 3 and ratio >= 0.35


# ==================================================
# PAYLOAD NORMALIZATION
# ==================================================

def build_candidate(point, distance=None):

    payload = point.payload or {}

    # Support both flat payloads and nested metadata.
    metadata = payload.get("metadata") or {}

    text = (
        payload.get("text")
        or payload.get("document")
        or payload.get("content")
        or ""
    )

    source = (
        payload.get("source")
        or payload.get("filename")
        or payload.get("document_name")
        or metadata.get("source")
        or "Unknown source"
    )

    page = (
        payload.get("page")
        or metadata.get("page")
        or "?"
    )

    combined_metadata = {
        **metadata,
        **{
            key: value
            for key, value in payload.items()
            if key not in {"text", "document", "content", "metadata"}
        }
    }

    return {
        "id": str(point.id),
        "text": str(text),
        "source": str(source),
        "page": page,
        "chunk_index": payload.get(
            "chunk_index",
            metadata.get("chunk_index", "?")
        ),
        "section": payload.get(
            "section",
            metadata.get("section", "general")
        ),
        "section_heading": payload.get(
            "section_heading",
            metadata.get("section_heading", "General")
        ),
        "metadata": combined_metadata,
        "distance": (
            float(distance)
            if distance is not None
            else None
        ),
        "vector_rank": None,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_score": 0.0,
        "rerank_score": None
    }


# ==================================================
# BM25 INDEX
# ==================================================

@lru_cache(maxsize=1)
def get_bm25_index():

    logger.info("Building BM25 index from Qdrant.")

    client = get_qdrant_client()

    corpus = []
    records = []

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:

            candidate = build_candidate(point)

            text = candidate["text"]

            if not text.strip():
                continue

            if (
                FILTER_REFERENCE_CHUNKS
                and is_reference_like(candidate)
            ):
                continue

            tokens = tokenize(text)

            if not tokens:
                continue

            corpus.append(tokens)
            records.append(candidate)

        if next_offset is None:
            break

        offset = next_offset

    if not corpus:

        logger.warning("No indexed documents available for BM25.")

        return None, []

    bm25 = BM25Okapi(corpus)

    logger.info("BM25 index ready with %s chunks.", len(records))

    return bm25, records


# ==================================================
# VECTOR SEARCH
# ==================================================

def vector_search(question, candidate_k):

    client = get_qdrant_client()

    embedder = get_embedder()

    embedding_start = perf_counter()

    vector = embedder.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    embedding_seconds = perf_counter() - embedding_start

    if len(vector) != VECTOR_SIZE:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {VECTOR_SIZE}, "
            f"received {len(vector)}."
        )

    search_start = perf_counter()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=candidate_k,
        with_payload=True,
        with_vectors=False
    )

    search_seconds = perf_counter() - search_start

    candidates = []

    for index, point in enumerate(response.points):

        candidate = build_candidate(
            point,
            distance=point.score
        )

        candidate["vector_rank"] = index + 1

        if not candidate["text"].strip():
            continue

        if (
            FILTER_REFERENCE_CHUNKS
            and is_reference_like(candidate)
        ):
            continue

        candidates.append(candidate)

    return candidates, embedding_seconds, search_seconds


# ==================================================
# BM25 KEYWORD SEARCH
# ==================================================

def keyword_search(question, candidate_k):

    bm25, records = get_bm25_index()

    if bm25 is None:
        return []

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    candidates = []

    for index in ranked_indices:

        score = float(scores[index])

        if score <= 0:
            continue

        candidate = dict(records[index])

        candidate["bm25_rank"] = len(candidates) + 1
        candidate["bm25_score"] = score

        candidates.append(candidate)

        if len(candidates) >= candidate_k:
            break

    return candidates


# ==================================================
# RECIPROCAL RANK FUSION
# ==================================================

def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    rrf_k=RRF_K
):

    fused = {}

    for ranking in (vector_results, keyword_results):

        for rank, candidate in enumerate(ranking, start=1):

            document_id = candidate["id"]

            if document_id not in fused:

                fused[document_id] = dict(candidate)
                fused[document_id]["rrf_score"] = 0.0

            fused[document_id]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )


# ==================================================
# CROSS-ENCODER RERANKING
# ==================================================

def rerank_candidates(question, candidates, top_k):

    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    for candidate, score in zip(candidates, scores):

        candidate["rerank_score"] = float(score)

    ranked = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]


# ==================================================
# MAIN RETRIEVAL PIPELINE
# ==================================================

def retrieve(
    question,
    top_k=DEFAULT_TOP_K,
    candidate_k=DEFAULT_CANDIDATE_K,
    use_reranker=True,
    return_timings=False,
    use_hybrid=True
):

    total_start = perf_counter()

    timings = {}

    if not question or not question.strip():

        if return_timings:
            return [], timings

        return []

    # 1. Vector retrieval
    vector_start = perf_counter()

    vector_results, embedding_seconds, vector_seconds = (
        vector_search(question, candidate_k)
    )

    timings["embedding_seconds"] = round(
        embedding_seconds, 4
    )

    timings["vector_search_seconds"] = round(
        vector_seconds, 4
    )

    timings["vector_retrieval_seconds"] = round(
        time_elapsed(vector_start), 4
    )

    # 2. BM25 retrieval
    keyword_start = perf_counter()

    if use_hybrid:
        keyword_results = keyword_search(
            question,
            candidate_k
        )
    else:
        keyword_results = []

    timings["keyword_search_seconds"] = round(
        time_elapsed(keyword_start), 4
    )

    # 3. Hybrid fusion
    fusion_start = perf_counter()

    if use_hybrid:

        candidates = reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )

    else:

        candidates = vector_results

    timings["fusion_seconds"] = round(
        time_elapsed(fusion_start), 4
    )

    # 4. Reranking
    rerank_start = perf_counter()

    if use_reranker and candidates:

        candidates = rerank_candidates(
            question,
            candidates,
            top_k
        )

    else:

        candidates = candidates[:top_k]

    timings["reranking_seconds"] = round(
        time_elapsed(rerank_start), 4
    )

    timings["retrieval_seconds"] = round(
        perf_counter() - total_start, 4
    )

    timings["vector_candidates"] = len(vector_results)
    timings["keyword_candidates"] = len(keyword_results)

    if return_timings:
        return candidates, timings

    return candidates


def time_elapsed(start):

    return perf_counter() - start


# ==================================================
# DIRECT TEST
# ==================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("\nConnecting to Qdrant...")

    info = check_collection()

    print("Qdrant connection successful.")
    print("Collection:", COLLECTION_NAME)
    print("Indexed points:", info["points_count"])

    if info["points_count"] == 0:

        print(
            "\nWARNING: Your collection is empty. "
            "Run the Qdrant ingestion script before testing retrieval."
        )

    else:

        question = input("\nEnter a test question: ").strip()

        results, timings = retrieve(
            question,
            top_k=3,
            candidate_k=20,
            use_reranker=True,
            return_timings=True
        )

        print("\nRETRIEVAL RESULTS")

        for index, result in enumerate(results, start=1):

            print(f"\n{index}. {result['source']}")
            print("Page:", result["page"])
            print("Score:", result["rerank_score"])
            print("Text:", result["text"][:500])

        print("\nTIMINGS")

        for key, value in timings.items():
            print(f"{key}: {value}")

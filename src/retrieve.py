
from pathlib import Path
from functools import lru_cache
from time import perf_counter

import os
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Set CHROMA_COLLECTION=rag_documents_v2 in PowerShell
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "rag_documents"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 3

# Retrieve a wider pool before reranking.
CANDIDATE_K = 20
RRF_K = 60
RERANK_CANDIDATE_LIMIT = 16

# Reference filtering
FILTER_REFERENCE_CHUNKS = True


# --------------------------------------------------
# Initialize ChromaDB and embedding model
# --------------------------------------------------

_db = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    _collection = _db.get_collection(COLLECTION_NAME)

except Exception as exc:
    available = [
        item.name
        for item in _db.list_collections()
    ]

    raise RuntimeError(
        f"ChromaDB collection '{COLLECTION_NAME}' was not found. "
        f"Available collections: {available}. "
        "Run the ingestion script first or check CHROMA_COLLECTION."
    ) from exc

_embedder = SentenceTransformer(EMBEDDING_MODEL)


# --------------------------------------------------
# Tokenization and normalization
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(text.lower().split())


# --------------------------------------------------
# Reference-content detection
# --------------------------------------------------

REFERENCE_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(?:references|bibliography|works cited)\s*$",
    re.IGNORECASE,
)

REFERENCE_LINE_PATTERNS = [
    re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"\bet al\.\s*\(?\d{4}\)?", re.IGNORECASE),
    re.compile(
        r"\b\d{4}\.\s*[A-Z][a-z]+.*\b"
        r"(?:proceedings|journal|conference)\b",
        re.IGNORECASE,
    ),
]


def is_reference_like(candidate: dict) -> bool:
    """
    Detect chunks that appear to contain bibliography entries.

    This uses both section metadata and content patterns.
    It deliberately avoids treating every URL or author name
    as a reference, since technical documentation may contain URLs.
    """

    metadata = candidate.get("metadata", {})

    section = str(
        metadata.get("section", "")
    ).lower().strip()

    heading = str(
        metadata.get("section_heading", "")
    ).lower().strip()

    if section in {
        "references",
        "bibliography",
        "works_cited",
    }:
        return True

    if heading in {
        "references",
        "bibliography",
        "works cited",
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

    # A standalone bibliography heading is a strong signal.
    if REFERENCE_SECTION_PATTERN.match(lines[0]):
        return True

    # Count lines that resemble formal academic references.
    reference_lines = 0

    for line in lines:

        has_year = bool(
            re.search(
                r"\b(?:19|20)\d{2}[a-z]?\b",
                line,
            )
        )

        has_author_pattern = bool(
            re.search(
                r"\b[A-Z][a-z]+,\s*(?:[A-Z]\.?|[A-Z][a-z]+)",
                line,
            )
        )

        has_reference_marker = any(
            pattern.search(line)
            for pattern in REFERENCE_LINE_PATTERNS
        )

        if has_reference_marker or (
            has_year and has_author_pattern
        ):
            reference_lines += 1

    reference_ratio = reference_lines / max(len(lines), 1)

    # Require several citation-like lines to avoid filtering
    # normal technical content containing a URL or date.
    return (
        reference_lines >= 3
        and reference_ratio >= 0.35
    )


def filter_reference_candidates(
    candidates: list[dict],
) -> list[dict]:
    """Remove likely bibliography chunks from retrieval results."""

    if not FILTER_REFERENCE_CHUNKS:
        return candidates

    return [
        candidate
        for candidate in candidates
        if not is_reference_like(candidate)
    ]


# --------------------------------------------------
# BM25 index
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_bm25_index():
    """
    Build a BM25 index from the existing ChromaDB collection.

    The index is cached in memory. Restart the Python process
    after changing or rebuilding the collection.
    """

    print("Building BM25 keyword index...")

    data = _collection.get(
        include=["documents", "metadatas"]
    )

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    corpus = []
    records = []

    for index, document in enumerate(documents):

        if not document or not document.strip():
            continue

        tokens = tokenize(document)

        if not tokens:
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        ) or {}

        record = {
            "id": (
                ids[index]
                if index < len(ids)
                else str(index)
            ),
            "text": document,
            "metadata": metadata,
        }

        # Do not add obvious bibliography chunks to BM25.
        if (
            FILTER_REFERENCE_CHUNKS
            and is_reference_like(record)
        ):
            continue

        corpus.append(tokens)
        records.append(record)

    if not corpus:
        print("No valid documents found for BM25.")
        return None, []

    bm25 = BM25Okapi(corpus)

    print(f"BM25 index ready: {len(records)} chunks.")

    return bm25, records


# --------------------------------------------------
# CrossEncoder reranker
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker():

    print("Loading CrossEncoder reranker...")

    model = CrossEncoder(RERANKER_MODEL)

    print("CrossEncoder loaded successfully.")

    return model


# --------------------------------------------------
# Candidate builder
# --------------------------------------------------

def build_candidate(
    document,
    metadata,
    document_id,
    distance=None,
):
    """Create a consistent candidate object."""

    metadata = metadata or {}

    return {
        "id": document_id,
        "text": document,
        "metadata": metadata,

        "source": metadata.get(
            "source",
            "Unknown source",
        ),

        "page": metadata.get("page", "?"),

        "chunk_index": metadata.get(
            "chunk_index",
            "?",
        ),

        "section": metadata.get(
            "section",
            "general",
        ),

        "section_heading": metadata.get(
            "section_heading",
            "General",
        ),

        "chunk_type": metadata.get(
            "chunk_type",
            "content",
        ),

        "distance": (
            float(distance)
            if distance is not None
            else None
        ),

        "vector_rank": None,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_score": 0.0,
        "rerank_score": None,
    }


# --------------------------------------------------
# Semantic vector search
# --------------------------------------------------

def vector_search(question, candidate_k):

    embedding_start = perf_counter()

    question_vector = _embedder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    embedding_seconds = (
        perf_counter() - embedding_start
    )

    search_start = perf_counter()

    results = _collection.query(
        query_embeddings=question_vector,
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    vector_search_seconds = (
        perf_counter() - search_start
    )

    documents = (
        results.get("documents") or [[]]
    )[0]

    metadatas = (
        results.get("metadatas") or [[]]
    )[0]

    distances = (
        results.get("distances") or [[]]
    )[0]

    ids = (
        results.get("ids") or [[]]
    )[0]

    candidates = []

    for index, document in enumerate(documents):

        if not document:
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        ) or {}

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        document_id = (
            ids[index]
            if index < len(ids)
            else f"vector-{index}"
        )

        candidate = build_candidate(
            document,
            metadata,
            document_id,
            distance,
        )

        candidate["vector_rank"] = index + 1

        candidates.append(candidate)

    return (
        candidates,
        embedding_seconds,
        vector_search_seconds,
    )


# --------------------------------------------------
# BM25 keyword search
# --------------------------------------------------

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
        reverse=True,
    )

    candidates = []

    for index in ranked_indices:

        score = float(scores[index])

        if score <= 0:
            continue

        record = records[index]

        candidate = build_candidate(
            record["text"],
            record["metadata"],
            record["id"],
        )

        candidate["bm25_rank"] = len(candidates) + 1
        candidate["bm25_score"] = score

        candidates.append(candidate)

        if len(candidates) >= candidate_k:
            break

    return candidates


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    vector_candidates,
    bm25_candidates,
):
    """
    Combine semantic and keyword rankings using RRF.

    Candidates are merged using their ChromaDB document IDs.
    Identical text from different documents or pages remains
    separate, preserving correct source attribution.
    """

    merged = {}

    # Add semantic vector rankings.
    for candidate in vector_candidates:

        key = candidate["id"]

        if key not in merged:
            merged[key] = candidate.copy()

        merged[key]["rrf_score"] += (
            1 / (
                RRF_K
                + candidate["vector_rank"]
            )
        )

    # Add BM25 keyword rankings.
    for candidate in bm25_candidates:

        key = candidate["id"]

        if key not in merged:
            merged[key] = candidate.copy()

        existing = merged[key]

        existing["rrf_score"] += (
            1 / (
                RRF_K
                + candidate["bm25_rank"]
            )
        )

        existing["bm25_rank"] = candidate["bm25_rank"]
        existing["bm25_score"] = candidate["bm25_score"]

    fused_candidates = list(merged.values())

    fused_candidates.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return fused_candidates


# --------------------------------------------------
# Retrieval diagnostics
# --------------------------------------------------

def create_empty_timings():
    """Create a consistent timings and diagnostics dictionary."""

    return {
        # Performance timings
        "embedding_seconds": 0.0,
        "vector_search_seconds": 0.0,
        "bm25_search_seconds": 0.0,
        "filtering_seconds": 0.0,
        "fusion_seconds": 0.0,
        "reranking_seconds": 0.0,
        "retrieval_seconds": 0.0,

        # Candidate diagnostics
        "vector_candidates_count": 0,
        "bm25_candidates_count": 0,
        "vector_candidates_after_filter": 0,
        "bm25_candidates_after_filter": 0,
        "fused_candidates_count": 0,
        "reranked_candidates_count": 0,
        "selected_candidates_count": 0,

        # Score and source diagnostics
        "top_rerank_score": None,
        "lowest_selected_rerank_score": None,
        "selected_sources": [],
    }


# --------------------------------------------------
# Main retrieval function
# --------------------------------------------------

def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = CANDIDATE_K,
    use_reranker: bool = True,
    return_timings: bool = False,
    use_hybrid: bool = True,
):
    """
    Hybrid Retrieval Pipeline:

    1. Semantic vector search.
    2. BM25 keyword search.
    3. Remove likely bibliography chunks.
    4. RRF fusion.
    5. CrossEncoder reranking.
    6. Return top relevant chunks.
    7. Record retrieval diagnostics.

    Return compatibility:
        retrieve(question) -> selected_candidates

        retrieve(question, return_timings=True)
            -> (selected_candidates, timings)
    """

    total_start = perf_counter()

    timings = create_empty_timings()

    if not question or not question.strip():
        return (
            ([], timings)
            if return_timings
            else []
        )

    if top_k < 1 or candidate_k < 1:
        raise ValueError(
            "top_k and candidate_k must be positive."
        )

    count = _collection.count()

    if count == 0:

        timings["retrieval_seconds"] = round(
            perf_counter() - total_start,
            4,
        )

        return (
            ([], timings)
            if return_timings
            else []
        )

    candidate_k = min(
        max(top_k, candidate_k),
        count,
    )

    clean_question = question.strip()

    # ----------------------------------------------
    # Step 1: Semantic vector search
    # ----------------------------------------------

    (
        vector_candidates,
        embedding_time,
        vector_time,
    ) = vector_search(
        clean_question,
        candidate_k,
    )

    timings["embedding_seconds"] = round(
        embedding_time,
        4,
    )

    timings["vector_search_seconds"] = round(
        vector_time,
        4,
    )

    timings["vector_candidates_count"] = len(
        vector_candidates
    )

    # ----------------------------------------------
    # Step 2: BM25 keyword search
    # ----------------------------------------------

    bm25_candidates = []

    if use_hybrid:

        bm25_start = perf_counter()

        bm25_candidates = keyword_search(
            clean_question,
            candidate_k,
        )

        timings["bm25_search_seconds"] = round(
            perf_counter() - bm25_start,
            4,
        )

    timings["bm25_candidates_count"] = len(
        bm25_candidates
    )

    # ----------------------------------------------
    # Step 3: Remove likely bibliography chunks
    # ----------------------------------------------

    filtering_start = perf_counter()

    vector_candidates = filter_reference_candidates(
        vector_candidates
    )

    bm25_candidates = filter_reference_candidates(
        bm25_candidates
    )

    timings["filtering_seconds"] = round(
        perf_counter() - filtering_start,
        4,
    )

    timings["vector_candidates_after_filter"] = len(
        vector_candidates
    )

    timings["bm25_candidates_after_filter"] = len(
        bm25_candidates
    )

    # ----------------------------------------------
    # Step 4: RRF fusion
    # ----------------------------------------------

    fusion_start = perf_counter()

    if use_hybrid:

        candidates = reciprocal_rank_fusion(
            vector_candidates,
            bm25_candidates,
        )

    else:

        candidates = vector_candidates

        for candidate in candidates:
            candidate["rrf_score"] = 0.0

    timings["fusion_seconds"] = round(
        perf_counter() - fusion_start,
        4,
    )

    timings["fused_candidates_count"] = len(
        candidates
    )

    # ----------------------------------------------
    # Step 5: Limit reranking workload
    # ----------------------------------------------

    if use_reranker:
        candidates = candidates[
            :RERANK_CANDIDATE_LIMIT
        ]

    timings["reranked_candidates_count"] = (
        len(candidates)
        if use_reranker
        else 0
    )

    # ----------------------------------------------
    # Step 6: CrossEncoder reranking
    # ----------------------------------------------

    reranking_start = perf_counter()

    if use_reranker and len(candidates) > 1:

        reranker = get_reranker()

        pairs = [
            (clean_question, item["text"])
            for item in candidates
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
            batch_size=8,
        )

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

    timings["reranking_seconds"] = round(
        perf_counter() - reranking_start,
        4,
    )

    # ----------------------------------------------
    # Step 7: Final results and diagnostics
    # ----------------------------------------------

    selected = candidates[
        :min(top_k, len(candidates))
    ]

    timings["selected_candidates_count"] = len(
        selected
    )

    # Record the selected reranker scores.
    selected_scores = [
        item["rerank_score"]
        for item in selected
        if item["rerank_score"] is not None
    ]

    if selected_scores:

        timings["top_rerank_score"] = round(
            max(selected_scores),
            4,
        )

        timings["lowest_selected_rerank_score"] = round(
            min(selected_scores),
            4,
        )

    # Record selected source information.
    timings["selected_sources"] = [
        {
            "source": item["source"],
            "page": item["page"],
            "chunk_index": item["chunk_index"],
            "rerank_score": (
                round(item["rerank_score"], 4)
                if item["rerank_score"] is not None
                else None
            ),
        }
        for item in selected
    ]

    timings["retrieval_seconds"] = round(
        perf_counter() - total_start,
        4,
    )

    if return_timings:
        return selected, timings

    return selected


# --------------------------------------------------
# Interactive terminal testing
# --------------------------------------------------

def main():

    print("\n========== RETRIEVAL CONFIGURATION ==========")

    print(f"Indexed chunks: {_collection.count()}")
    print(f"Chroma collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranker model: {RERANKER_MODEL}")
    print("Hybrid search: enabled")
    print(f"Candidate pool: {CANDIDATE_K}")
    print(f"Reranker limit: {RERANK_CANDIDATE_LIMIT}")
    print(f"Reference filtering: {FILTER_REFERENCE_CHUNKS}")

    print("\nEnter 'exit' to stop.")

    while True:

        try:

            question = input(
                "\nAsk a question about your documents: "
            ).strip()

        except (EOFError, KeyboardInterrupt):

            print("\nExiting retrieval tester.")
            break

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            results, timings = retrieve(
                question,
                top_k=3,
                candidate_k=CANDIDATE_K,
                use_reranker=True,
                return_timings=True,
                use_hybrid=True,
            )

        except Exception as exc:

            print(f"Retrieval failed: {exc}")
            continue

        # ------------------------------------------
        # Retrieved chunks
        # ------------------------------------------

        print("\n========== RETRIEVED CHUNKS ==========")

        if not results:
            print("No relevant chunks were retrieved.")

        for index, item in enumerate(results, start=1):

            print(f"\nResult {index}")

            print(
                f"Source: {item['source']} | "
                f"Page: {item['page']} | "
                f"Chunk: {item['chunk_index']}"
            )

            print(
                f"Section: {item['section']} | "
                f"Heading: {item['section_heading']} | "
                f"Type: {item['chunk_type']}"
            )

            print(f"Document ID: {item['id']}")
            print(f"Vector distance: {item['distance']}")
            print(f"Vector rank: {item['vector_rank']}")
            print(f"BM25 rank: {item['bm25_rank']}")
            print(f"BM25 score: {item['bm25_score']}")
            print(f"RRF score: {item['rrf_score']:.6f}")

            if item["rerank_score"] is not None:

                print(
                    f"Reranker score: "
                    f"{item['rerank_score']:.4f}"
                )

            print(f"Text: {item['text'][:1000]}")

        # ------------------------------------------
        # Performance timings
        # ------------------------------------------

        print("\n========== PERFORMANCE TIMINGS ==========")

        timing_keys = [
            "embedding_seconds",
            "vector_search_seconds",
            "bm25_search_seconds",
            "filtering_seconds",
            "fusion_seconds",
            "reranking_seconds",
            "retrieval_seconds",
        ]

        for name in timing_keys:
            print(f"{name}: {timings[name]:.4f}s")

        # ------------------------------------------
        # Candidate diagnostics
        # ------------------------------------------

        print("\n========== CANDIDATE DIAGNOSTICS ==========")

        diagnostic_keys = [
            "vector_candidates_count",
            "bm25_candidates_count",
            "vector_candidates_after_filter",
            "bm25_candidates_after_filter",
            "fused_candidates_count",
            "reranked_candidates_count",
            "selected_candidates_count",
        ]

        for name in diagnostic_keys:
            print(f"{name}: {timings[name]}")

        # ------------------------------------------
        # Score diagnostics
        # ------------------------------------------

        print("\n========== SCORE DIAGNOSTICS ==========")

        print(
            "Top reranker score: "
            f"{timings['top_rerank_score']}"
        )

        print(
            "Lowest selected reranker score: "
            f"{timings['lowest_selected_rerank_score']}"
        )

        # ------------------------------------------
        # Selected sources
        # ------------------------------------------

        print("\n========== SELECTED SOURCES ==========")

        for source in timings["selected_sources"]:

            print(
                f"{source['source']} | "
                f"Page {source['page']} | "
                f"Chunk {source['chunk_index']} | "
                f"Score {source['rerank_score']}"
            )


if __name__ == "__main__":
    main()

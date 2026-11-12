
import os
import time
import uuid
import logging
import re

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from src.retrieve import retrieve, get_qdrant_client, COLLECTION_NAME


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

TOP_K = int(os.getenv("TOP_K", "3"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "2048")
)

THINKING_LEVEL = os.getenv(
    "THINKING_LEVEL",
    "LOW"
).upper()

API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=API_KEY)
    if API_KEY
    else None
)

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="AtlasQuery API",
    description="Qdrant-powered Retrieval Augmented Generation API",
    version="3.1.0"
)

# ==================================================
# CORS CONFIGURATION
# ==================================================

origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# REQUEST MODELS
# ==================================================

class HistoryMessage(BaseModel):

    role: str

    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=4000
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=12
    )


# ==================================================
# HISTORY
# ==================================================

def build_history(history):

    messages = []

    for message in history[-12:]:

        role = message.role.lower()

        if role not in ("user", "assistant"):
            continue

        messages.append(
            f"{role.upper()}: {message.content}"
        )

    return "\n".join(messages)


def contextualize_question(question, history):

    if not history:
        return question

    previous_user_questions = [
        item.content
        for item in history
        if item.role.lower() == "user"
    ]

    if not previous_user_questions:
        return question

    standalone_markers = {
        "it", "they", "them", "this",
        "that", "these", "those",
        "its", "their"
    }

    words = question.lower().split()

    needs_context = (
        len(words) <= 8
        or any(word in standalone_markers for word in words)
    )

    if not needs_context:
        return question

    previous_question = previous_user_questions[-1]

    return (
        f"Previous user question: {previous_question}\n"
        f"Current follow-up question: {question}"
    )


# ==================================================
# SOURCE FORMATTING
# ==================================================

def build_context(results):

    context_parts = []
    sources = []

    for index, result in enumerate(results, start=1):

        source_id = f"S{index}"

        source_name = result.get(
            "source",
            "Unknown"
        )

        page = result.get("page")

        text = result.get("text", "")

        context_parts.append(
            f"[{source_id}]\n"
            f"Document: {source_name}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

        sources.append({
            "id": source_id,
            "source": source_name,
            "name": source_name,
            "page": page,
            "text": text,
            "distance": result.get("distance"),
            "rerank_score": result.get("rerank_score")
        })

    return "\n\n".join(context_parts), sources


# ==================================================
# PROMPT
# ==================================================

def build_prompt(question, history, context):

    return f"""
You are AtlasQuery, a document-grounded AI research assistant.

Your job is to answer questions using the supplied document passages.

STRICT RULES:

1. Use the supplied document passages as factual evidence.
2. Do not invent facts, statistics, definitions or references.
3. If the documents do not contain enough information, clearly
   state that the available documents do not provide sufficient
   evidence to answer.
4. Cite factual statements using source labels such as [S1].
5. Do not create citations that are not present in the context.
6. Use clear headings and structured explanations when useful.
7. If the user asks a general question unrelated to the documents,
   do not pretend that the documents support the answer.
8. Conversation history can help interpret follow-up questions,
   but it is not a source of factual evidence.

CONVERSATION HISTORY:

{history if history else "No previous conversation."}

RETRIEVED DOCUMENT PASSAGES:

{context}

USER QUESTION:

{question}

Provide an accurate, well-structured response with citations.
"""


# ==================================================
# CITATION VALIDATION
# ==================================================

def validate_citations(answer, sources):

    valid_ids = {
        source["id"]
        for source in sources
    }

    citations = re.findall(
        r"\[(S\d+)\]",
        answer
    )

    invalid = [
        citation
        for citation in citations
        if citation not in valid_ids
    ]

    return {
        "valid": len(invalid) == 0,
        "invalid_citations": invalid
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/")
def root():

    return {
        "name": "AtlasQuery API",
        "version": "3.1.0",
        "status": "running",
        "database": "Qdrant Cloud"
    }


@app.get("/api/health")
def health():

    qdrant_status = "disconnected"
    collection_count = 0
    collection_status = "unknown"
    qdrant_error = None

    try:

        client = get_qdrant_client()

        info = client.get_collection(
            COLLECTION_NAME
        )

        collection_count = info.points_count or 0

        collection_status = str(info.status)

        qdrant_status = "connected"

    except Exception as error:

        logger.exception("Qdrant health check failed.")

        qdrant_error = str(error)

    gemini_configured = gemini_client is not None

    fully_ready = (
        qdrant_status == "connected"
        and collection_count > 0
        and gemini_configured
    )

    return {
        # Frontend expects "ok".
        "status": "ok" if fully_ready else "degraded",

        "version": "3.1.0",

        "database": {
            "provider": "Qdrant Cloud",
            "status": qdrant_status,
            "collection": COLLECTION_NAME,
            "indexed_chunks": collection_count,
            "collection_status": collection_status,
            "error": qdrant_error
        },

        "gemini": {
            "configured": gemini_configured,
            "model": MODEL_NAME
        }
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    request_id = str(uuid.uuid4())

    start_total = time.perf_counter()

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if gemini_client is None:

        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )

    try:

        # 1. Contextualize the question.

        retrieval_query = contextualize_question(
            question,
            request.history
        )

        history_text = build_history(
            request.history
        )

        # 2. Retrieve documents from Qdrant.

        retrieval_start = time.perf_counter()

        results, retrieval_timings = retrieve(
            retrieval_query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            use_reranker=True,
            return_timings=True,
            use_hybrid=True
        )

        retrieval_seconds = (
            time.perf_counter() - retrieval_start
        )

        # 3. Handle missing evidence.

        if not results:

            total_seconds = (
                time.perf_counter() - start_total
            )

            return {
                "request_id": request_id,

                "answer": (
                    "I could not find relevant passages "
                    "in the indexed documents to answer "
                    "your question reliably."
                ),

                "sources": [],

                "warnings": [
                    "No relevant document passages were retrieved."
                ],

                "latency_seconds": round(total_seconds, 3),

                "timings": {
                    **retrieval_timings,
                    "retrieval_seconds": round(
                        retrieval_seconds, 3
                    ),
                    "generation_seconds": 0,
                    "total_seconds": round(
                        total_seconds, 3
                    )
                },

                "token_usage": None,
                "finish_reason": None,
                "evidence_status": "insufficient"
            }

        # 4. Build evidence context.

        context, sources = build_context(results)

        prompt = build_prompt(
            question,
            history_text,
            context
        )

        # 5. Generate answer using Gemini.

        generation_start = time.perf_counter()

        config = types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=THINKING_LEVEL
            )
        )

        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )

        generation_seconds = (
            time.perf_counter() - generation_start
        )

        answer = response.text or (
            "I could not generate a response. Please try again."
        )

        # 6. Validate citations.

        citation_check = validate_citations(
            answer,
            sources
        )

        warnings = []

        if not citation_check["valid"]:

            warnings.append(
                "The response contains citation labels "
                "that were not found in the retrieved sources."
            )

        # 7. Token usage.

        usage = getattr(
            response,
            "usage_metadata",
            None
        )

        token_usage = None

        if usage:

            token_usage = {
                "prompt_tokens": getattr(
                    usage,
                    "prompt_token_count",
                    None
                ),

                "completion_tokens": getattr(
                    usage,
                    "candidates_token_count",
                    None
                ),

                "total_tokens": getattr(
                    usage,
                    "total_token_count",
                    None
                )
            }

        # 8. Finish reason.

        finish_reason = None

        candidates = getattr(
            response,
            "candidates",
            None
        )

        if candidates:

            finish_reason = str(
                getattr(
                    candidates[0],
                    "finish_reason",
                    None
                )
            )

        # 9. Evidence status.

        evidence_status = (
            "supported"
            if citation_check["valid"]
            else "citation_warning"
        )

        # 10. Timing.

        total_seconds = (
            time.perf_counter() - start_total
        )

        timings = {
            **retrieval_timings,

            "retrieval_seconds": round(
                retrieval_seconds, 3
            ),

            "generation_seconds": round(
                generation_seconds, 3
            ),

            "total_seconds": round(
                total_seconds, 3
            )
        }

        return {
            "request_id": request_id,
            "answer": answer,
            "sources": sources,
            "warnings": warnings,
            "latency_seconds": round(total_seconds, 3),
            "timings": timings,
            "token_usage": token_usage,
            "finish_reason": finish_reason,
            "evidence_status": evidence_status
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Chat request failed. Request ID: %s",
            request_id
        )

        error_text = str(error).lower()

        if "429" in error_text or "resource_exhausted" in error_text:

            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exceeded. Please try again later."
            )

        if "503" in error_text or "unavailable" in error_text:

            raise HTTPException(
                status_code=503,
                detail="Gemini service is temporarily unavailable."
            )

        raise HTTPException(
            status_code=500,
            detail=f"AtlasQuery encountered an internal error. Request ID: {request_id}"
        )

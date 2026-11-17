# AtlasQuery – AI-Powered Multi-Document PDF RAG Assistant

**AtlasQuery** is an AI-powered document question-answering platform that allows users to upload PDF documents and ask questions about their contents using natural language.

It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-aware answers using Google's Gemini API.

Instead of relying only on an LLM's general knowledge, AtlasQuery retrieves relevant document passages and uses them as context when generating responses.

## Live Demo

- **Frontend:** https://atlas-query.vercel.app/
- **Backend API:** https://atlas-query.onrender.com

## Features

- **Multi-PDF Upload:** Upload multiple PDF documents to build a searchable document collection.
- **PDF Text Extraction:** Extract document content using PyMuPDF.
- **Text Chunking:** Divide extracted content into smaller chunks for efficient retrieval.
- **Semantic Search:** Generate document embeddings using FastEmbed.
- **Vector Database:** Store and retrieve embeddings using Qdrant Cloud.
- **Hybrid Retrieval:** Combine semantic vector search and BM25 keyword search.
- **Reciprocal Rank Fusion:** Combine retrieval results to improve the relevance of retrieved passages.
- **Context-Aware Answers:** Use retrieved document content as context for Gemini-generated responses.
- **Document-Specific Retrieval:** Filter retrieval results using document identifiers.
- **Web-Based Interface:** Interact with uploaded documents through a React application.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| LLM | Google Gemini API |
| Embedding Model | FastEmbed – all-MiniLM-L6-v2 |
| Vector Database | Qdrant Cloud |
| PDF Processing | PyMuPDF |
| Keyword Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |

## System Architecture

```text
                 USER
                   |
                   v
          React + Vite Frontend
                   |
                   v
             FastAPI Backend
                   |
          +--------+--------+
          |                 |
          v                 v
      PDF Upload        User Question
          |                 |
          v                 v
      PyMuPDF          Query Embedding
          |                 |
          v                 v
     Text Extraction    Qdrant Search
          |                 |
          v                 v
      Text Chunking     BM25 Search
          |                 |
          v                 v
    FastEmbed Model    Reciprocal Rank
          |              Fusion
          v                 |
     Qdrant Cloud            |
          |                 |
          +--------+--------+
                   |
                   v
           Relevant Context
                   |
                   v
             Gemini API
                   |
                   v
          Context-Aware Answer
                   |
                   v
             User Interface
```

## How It Works

### 1. Document Upload

The user uploads a PDF through the frontend. The FastAPI backend receives the file and validates it before processing.

### 2. Text Extraction

PyMuPDF extracts text from the PDF, preserving page-level information for document references.

### 3. Text Chunking

The extracted content is divided into smaller chunks so that relevant passages can be retrieved efficiently.

### 4. Embedding Generation

FastEmbed generates 384-dimensional vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are generated through a shared model instance to avoid unnecessary duplicate model loading within the backend process.

### 5. Vector Storage

The generated embeddings and associated metadata are stored in Qdrant Cloud.

Stored metadata includes information such as:

- Document ID
- Source filename
- Page number
- Chunk index
- Extracted text

### 6. Hybrid Retrieval

When a user submits a question, AtlasQuery retrieves relevant document passages using:

- Semantic vector search through Qdrant.
- BM25 keyword-based retrieval.

The results are combined using **Reciprocal Rank Fusion (RRF)** to produce a more relevant set of passages.

### 7. Response Generation

The retrieved passages are provided as context to Google's Gemini API. Gemini uses the supplied document context to generate a natural-language response.

## Retrieval-Augmented Generation (RAG)

AtlasQuery follows a document-grounded RAG workflow:

1. Receive a user question.
2. Retrieve relevant document chunks.
3. Combine semantic and keyword search results.
4. Construct a context from the retrieved passages.
5. Send the question and context to Gemini.
6. Return the generated response to the frontend.

This approach helps the system answer questions using uploaded documents rather than relying exclusively on general LLM knowledge.

## Project Structure

```text
AtlasQuery/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── app.css
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── embedding.py
│   │   ├── ingest.py
│   │   └── retrieve.py
│   │
│   ├── requirements.txt
│   └── ...
│
└── README.md
```

*Note: Adjust the directory names above if your repository uses a different frontend/backend folder structure.*

## Environment Variables

The application requires environment variables for connecting to external services.

### Backend

Create a `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model

QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=atlasquery_documents

EMBEDDING_MODEL=all-MiniLM-L6-v2
FRONTEND_ORIGINS=http://localhost:5173
```

### Frontend

Configure the frontend environment:

```env
VITE_API_URL=http://localhost:8000
```

**Important:** Never commit API keys or `.env` files to GitHub. Configure production environment variables directly through the respective deployment dashboards.

## Local Setup

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- Google Gemini API key
- Qdrant Cloud account

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AtlasQuery
```

### 2. Set Up the Backend

```bash
cd backend

python -m venv venv
```

Activate the virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the backend environment variables and start the server:

```bash
uvicorn src.api:app --reload
```

The backend will run at:

`http://localhost:8000`

### 3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at:

`http://localhost:5173`

## Deployment

AtlasQuery is deployed using:

| Service | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render |
| Vector Database | Qdrant Cloud |
| LLM API | Google Gemini |

### Frontend Deployment

The React application is deployed on Vercel.

Configure:

```env
VITE_API_URL=https://atlas-query.onrender.com
```

### Backend Deployment

The FastAPI backend is deployed on Render.

Start command:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Configure the required Gemini, Qdrant, embedding model, and frontend-origin environment variables in the Render dashboard.

## Performance and Resource Optimization

The backend is designed to reduce unnecessary memory consumption, particularly when operating on resource-limited hosting.

Current optimizations include:

- Shared FastEmbed model initialization using `lru_cache`.
- Single-threaded embedding execution.
- Smaller embedding batches during document ingestion.
- Incremental PDF page processing.
- Cached external database client initialization.
- Disabled reranking to reduce additional model memory usage.
- Remote vector storage through Qdrant Cloud.

These optimizations reduce duplicate initialization and unnecessary memory overhead. Actual resource usage still depends on document size, request concurrency, and hosting limits.

## Limitations

- PDF text extraction depends on the availability of machine-readable text. Scanned PDFs may require OCR support.
- Response quality depends on document content, retrieval relevance, and LLM output.
- Gemini API usage is subject to model availability and API quota limits.
- Render's free hosting environment has limited memory and may experience cold starts or temporary service interruptions.
- The current implementation prioritizes lightweight retrieval over memory-intensive reranking.

## Future Enhancements

- OCR support for scanned PDFs.
- Improved document citation and source highlighting.
- Streaming AI responses.
- Advanced retrieval optimization.
- Optional reranking when sufficient resources are available.
- Conversation history and persistent chat sessions.
- Document summarization and comparison.
- Expanded support for document formats.

## Author

**Sahil Shete**

Developed as an AI-powered document intelligence project using Retrieval-Augmented Generation, semantic search, hybrid retrieval, and large language models.

## License

Add your preferred open-source license before distributing the project. If no license is selected, the repository remains under the default copyright terms.

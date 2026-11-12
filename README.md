# AtlasQuery – AI-Powered Document Intelligence Platform

**Ask questions. Explore documents. Discover what matters.**

AtlasQuery is an AI-powered document intelligence platform that allows users to interact with a collection of research papers and documents through natural language conversations.

Built using **Retrieval-Augmented Generation (RAG)**, AtlasQuery combines semantic search, keyword-based retrieval, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking to retrieve relevant document passages before generating answers using Google's Gemini API.

Unlike a conventional AI chatbot, AtlasQuery grounds its responses in retrieved document evidence and provides source references to help users verify the information.

---

## ✨ Key Features

### 1. AI-Powered Document Chat
- Ask natural language questions about indexed documents.
- Generate contextual responses using Gemini.
- Maintain conversation history for follow-up questions.
- Handle questions requiring information from multiple document passages.

### 2. Hybrid Retrieval System
AtlasQuery combines multiple retrieval techniques to improve the relevance of retrieved information.

- **Semantic Search:** Uses ChromaDB vector search to identify conceptually relevant passages.
- **Keyword Search:** Uses BM25 to identify relevant terms and phrases.
- **Reciprocal Rank Fusion:** Combines rankings from multiple retrieval methods.
- **Cross-Encoder Reranking:** Reorders retrieved passages based on their relevance to the original query.

### 3. Evidence-Grounded Responses
- Answers are generated using retrieved document context.
- Source references identify the documents and pages used.
- Evidence-aware prompting helps reduce unsupported responses.
- Handles questions when sufficient document evidence is unavailable.

### 4. Interactive Evidence Constellation
Explore the evidence behind generated answers through an interactive visualization.

- Visual representation of retrieved document sources.
- Interactive source nodes.
- Inspect retrieved text passages.
- View document names and page references.
- Navigate between evidence sources.

### 5. RAG Research Pipeline Visualization
Understand how a question moves through the retrieval and generation workflow.

- Question processing
- Hybrid retrieval
- Evidence reranking
- Gemini response generation

### 6. Research Activity Metrics
View available system metrics associated with generated responses.

- Retrieval time
- Generation time
- Total response time
- Number of retrieved evidence sources
- Evidence status, when provided by the backend

### 7. Modern Chat Interface
- Responsive React-based interface.
- Markdown response rendering.
- Loading indicators during AI generation.
- Visible API and connection error handling.
- Source inspection and answer copying.
- Recent conversation activity.

---

## 🧠 System Architecture

AtlasQuery follows a Retrieval-Augmented Generation architecture.

```text
                 USER QUESTION
                       |
                       v
              FASTAPI BACKEND
                       |
                       v
              QUERY PROCESSING
                       |
                       v
              HYBRID RETRIEVAL
                 /         \
                /           \
               v             v
        SEMANTIC SEARCH   KEYWORD SEARCH
          (ChromaDB)         (BM25)
               \             /
                \           /
                 v         v
             RECIPROCAL RANK FUSION
                       |
                       v
              CROSS-ENCODER
                RERANKING
                       |
                       v
              RETRIEVED CONTEXT
                       |
                       v
             EVIDENCE-AWARE PROMPT
                       |
                       v
                 GEMINI API
                       |
                       v
              GENERATED ANSWER
                       |
                       v
             SOURCES + RESPONSE
                       |
                       v
               REACT FRONTEND
```

### Document Processing Pipeline

Documents are processed before they become available for retrieval.

```text
PDF DOCUMENTS
      |
      v
   TEXT EXTRACTION
      |
      v
   TEXT CLEANING
      |
      v
   DOCUMENT CHUNKING
      |
      v
   TEXT EMBEDDINGS
      |
      v
   CHROMADB INDEXING
      |
      v
   RETRIEVAL READY
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js |
| Build Tool | Vite |
| Styling | CSS |
| Markdown Rendering | React Markdown |
| Backend | Python, FastAPI |
| LLM | Google Gemini API |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Reranker Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF Processing | Python PDF processing pipeline |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |

---

## 📂 Project Structure

```text
AtlasQuery/
│
├── data/
│   ├── research_paper_1.pdf
│   ├── research_paper_2.pdf
│   └── retrieval_eval.json
│
├── chroma_db/
│   └── Persistent vector database
│
├── src/
│   ├── api.py
│   ├── ingest.py
│   ├── retrieve.py
│   └── generate.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── .env
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

*Note: The structure above describes the main application components. Adjust filenames if your repository uses a different layout.*

---

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or later
- Node.js and npm
- Git
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AtlasQuery
```

### 2. Backend Setup

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### 3. Configure Backend Environment Variables

Create a `.env` file in the backend root directory.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model

THINKING_LEVEL=your_configured_thinking_level

TOP_K=3
CANDIDATE_K=20

CHROMA_COLLECTION=rag_documents_v2

FRONTEND_ORIGINS=http://localhost:5173
```

Replace the example values with the configuration expected by your backend.

**Important:** Never commit your `.env` file or expose your Gemini API key in frontend code.

### 4. Prepare the Document Database

Place your PDF documents inside the `data/` directory.

Run the ingestion pipeline:

```powershell
python -m src.ingest
```

This processes the documents, creates text chunks, generates embeddings, and stores them in ChromaDB.

Make sure the configured ChromaDB collection matches the collection created during ingestion.

### 5. Start the Backend

```powershell
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/api/health
```

### 6. Frontend Setup

Open another terminal:

```powershell
cd frontend
npm install
```

Create a frontend `.env` file:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## 🔍 How RAG Works in AtlasQuery

AtlasQuery uses a multi-stage retrieval pipeline rather than relying exclusively on a language model's pretrained knowledge.

### Stage 1: Document Chunking

PDF content is extracted, cleaned, and divided into smaller text chunks with overlapping context.

### Stage 2: Embedding Generation

Each document chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

These embeddings are stored in ChromaDB.

### Stage 3: Hybrid Retrieval

When a user submits a question, AtlasQuery performs two retrieval operations:

1. Semantic retrieval using ChromaDB.
2. Keyword retrieval using BM25.

This allows the system to identify both conceptually related passages and passages containing important keywords.

### Stage 4: Reciprocal Rank Fusion

The results from both retrieval methods are combined using Reciprocal Rank Fusion.

This helps consolidate different retrieval rankings into a unified candidate list.

### Stage 5: Cross-Encoder Reranking

A Cross-Encoder evaluates the relevance between the original question and retrieved passages.

The passages are reranked so that more relevant evidence can be prioritized.

### Stage 6: Grounded Response Generation

The selected passages are provided to Gemini through an engineered prompt.

The prompt instructs the model to:
- Answer using the supplied document context.
- Avoid inventing unsupported information.
- Reference relevant sources.
- Acknowledge when the available evidence is insufficient.

---

## 🧪 Retrieval Evaluation

AtlasQuery includes a retrieval evaluation dataset to test whether the retrieval pipeline identifies relevant document passages.

The evaluation covers different question categories:

- Conceptual questions
- Direct information retrieval
- Out-of-scope questions

Previously recorded evaluation results:

| Metric | Result |
|---|---:|
| Hit Rate@3 | 88.89% |
| Precision@3 | 51.85% |
| Mean Reciprocal Rank (MRR) | 0.7593 |

These results are from the project's earlier retrieval evaluation run and should not be interpreted as a guarantee of answer accuracy.

Run the retrieval evaluation workflow using your evaluation script and the configured ChromaDB collection to reproduce the results.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend root endpoint |
| GET | `/api/health` | Check backend and retrieval health |
| POST | `/api/chat` | Submit a question and receive an AI response |

### Example Chat Request

```json
{
  "question": "What are the major findings discussed in the research papers?",
  "history": []
}
```

### Example Response Structure

```json
{
  "request_id": "example-request-id",
  "answer": "The retrieved documents discuss...",
  "sources": [
    {
      "id": "S1",
      "source": "research_paper_1.pdf",
      "page": 5,
      "text": "Retrieved document passage..."
    }
  ],
  "warnings": [],
  "latency_seconds": 2.45,
  "timings": {
    "retrieval_seconds": 0.85,
    "generation_seconds": 1.35,
    "total_seconds": 2.45
  },
  "evidence_status": "supported"
}
```

*The response is illustrative. Actual fields and values depend on the backend implementation and generated response.*

---

## 🚀 Deployment

AtlasQuery uses separate frontend and backend deployments.

### Frontend – Vercel

1. Push the frontend code to GitHub.
2. Import the repository into Vercel.
3. Configure the frontend root directory as `frontend`.
4. Set the environment variable:

```env
VITE_API_URL=https://your-render-backend-url.onrender.com
```

5. Deploy the application.

### Backend – Render

1. Create a new Render Web Service.
2. Connect your GitHub repository.
3. Configure the Python environment.
4. Set the build command:

```bash
pip install -r requirements.txt
```

5. Set the start command:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

6. Add the required environment variables in Render.
7. Configure CORS to allow your deployed Vercel domain.
8. Configure persistent storage for ChromaDB or another reliable database initialization strategy.

**Deployment note:** Ensure that the deployed backend can access the correct indexed ChromaDB collection. A local database directory should not be assumed to persist automatically on a cloud hosting service.

---

## 🔐 Security

AtlasQuery follows basic API security practices:

- API keys are stored in backend environment variables.
- Gemini credentials are never exposed through frontend code.
- `.env` files are excluded from Git.
- Frontend requests communicate with the backend through API endpoints.
- Production CORS is configured for the deployed frontend domain.

---

## 🎯 Example Questions to Test

Use questions whose answers require information from your indexed documents.

| Question Type | Example |
|---|---|
| Conceptual | What are the main concepts explained in the research papers? |
| Direct retrieval | What methodology does the document describe? |
| Comparison | Compare the approaches discussed in the indexed papers. |
| Research findings | What are the major findings and conclusions? |
| Detailed analysis | Explain the limitations identified by the authors. |
| Out-of-scope | What is the capital of Japan? |

For a meaningful RAG evaluation, use at least one question whose answer is specific to your document collection and cannot be reliably answered from general knowledge alone.

---

## 🔮 Future Improvements

- Support for additional document formats.
- Document-level filtering and search.
- More advanced conversational memory.
- Improved retrieval evaluation and relevance calibration.
- Additional embedding and reranking models.
- User authentication and personal document collections.
- Exporting research answers and citations.
- More detailed retrieval observability.

---

## 👨‍💻 Project Information

**Project:** AtlasQuery – AI-Powered Document Intelligence Platform

**Core Concepts:**
- Large Language Models
- Prompt Engineering
- Retrieval-Augmented Generation
- Hybrid Information Retrieval
- Vector Embeddings
- Semantic Search
- BM25
- Reciprocal Rank Fusion
- Cross-Encoder Reranking

**Developed using:** React, FastAPI, ChromaDB, Python, and Google Gemini.

---

## 📄 License

This project is developed for educational and demonstration purposes.

Add an appropriate open-source license if you intend to distribute the project under specific reuse terms.

---

**AtlasQuery — Turning documents into discoverable knowledge.**
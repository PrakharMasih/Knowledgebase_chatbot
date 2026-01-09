# Codebase Walkthrough (Current Implementation)

## Technical Overview

This is a **Knowledge-Grounded RAG (Retrieval-Augmented Generation) Chatbot** built with FastAPI (Python) that answers user queries strictly based on PDF documents. The system implements a complete pipeline from document ingestion to conversational AI responses with chat history support.

## System Architecture

### 1. **Project Structure**

```
app/
├── config/
│   ├── settings.py          # Configuration & environment variables
│   ├── database.py          # SQLite async database setup
│   └── init_db.py           # Database initialization
├── core/
│   ├── embeddings.py        # Sentence transformer embedding function
│   ├── vector_store.py      # ChromaDB vector database operations
│   ├── document_processor.py # PDF loading and chunking
│   └── chat_history.py      # Conversation history management
├── chatbot/
│   ├── models.py            # SQLAlchemy database models
│   ├── schemas.py           # Pydantic request/response models
│   ├── routes.py            # FastAPI API endpoints
│   └── chat_service.py      # Core RAG query processing logic
└── main.py                  # Application entry point
```

## App StartUp Flow

<img width="1467" height="6764" alt="startup_flow" src="https://github.com/user-attachments/assets/de7197e0-c284-4d1f-9f45-661eb5b1b44e" />


## 2. High-Level Architecture (Startup Phase)

At server startup, the application performs the following high-level steps:

1. FastAPI application boots
2. Vector database is initialized
3. Relational database schema is created
4. PDF documents are loaded from disk
5. PDFs are chunked and embedded
6. Embeddings are stored in the vector database
7. Application enters ready state

---

## 3. Entry Point: `main.py`

### 3.1 Application Lifecycle Management

The application uses **FastAPI’s `lifespan` context manager** to control startup and shutdown logic.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```

This ensures:

* Startup logic runs **once** when the server starts
* Shared resources are stored in `app.state`
* Graceful shutdown logging is possible

---

### 3.2 Startup Responsibilities

#### Step 1: Vector Store Initialization

```python
vector_store = EnhancedVectorStore()
app.state.vector_store = vector_store
```

* Initializes the vector database (ChromaDB internally)
* Holds:

  * Embedding function - **ChromaDB SentenceTransformerEmbeddingFunction function is used ( model - all-MiniLM-L6-v2 )**
  * Persistent storage path (assumed)
* Stored in `app.state` for global access across requests

---

#### Step 2: Database Initialization

```python
await init_db()
```

* Uses `aiosqlite` (async SQLAlchemy engine)
* Creates tables defined in `Base.metadata`
* Typically used for:

  * Message

---

#### Step 3: Document Processor Initialization

```python
doc_processor = DocumentProcessor(vector_store)
```

* Binds document ingestion directly to the vector store
* Couples ingestion and embedding tightly 

---

#### Step 4: PDF Ingestion on Startup

```python
num_docs = await doc_processor.load_pdfs_from_directory(settings.PDF_DIRECTORY)
```

This step:

* Scans the configured data directory
* Processes **all PDFs synchronously during startup**
* Blocks server readiness until completion


---

## 4. Document Processing Pipeline (`document_processor.py`)

### 4.1 PDF Discovery

```python
pdf_files = list(directory.glob("*.pdf"))
```

* Only `.pdf` files are supported

---

### 4.2 PDF Loading

```python
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()
```

* Extracts text page-by-page
* Each page becomes a LangChain `Document`
* Metadata includes page index

---

### 4.3 Text Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)
```

* Character-based chunking
* Recursive fallback to smaller separators


---

### 4.4 Chunk Deduplication Logic

```python
chunk_id = md5(f"{filename}_{chunk_index}")
```

* Deterministic ID per file + chunk index
* Prevents re-ingesting same PDF **only if chunk order remains identical**

---

### 4.5 Vector Store Ingestion

```python
self.vector_store.add_documents(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
```

Each chunk stores:

* Text content
* Metadata:

  * source (filename)
  * page number
  * chunk index
* Vector embedding (generated internally)

📌 Metadata is minimal but sufficient for:

* Source attribution
* Page-level citations

---

## 5. Database Initialization (`init_db.py`)

```python
await conn.run_sync(Base.metadata.create_all)
```

* Uses SQLAlchemy async engine
* Ensures tables exist
* No migration/versioning strategy

Assumed use cases:

* Chat history persistence
* Session memory
* Feedback tracking (future)

---

## 6. System State After Startup

Once startup completes:

| Component    | Status                      |
| ------------ | --------------------------- |
| Vector Store | Initialized & populated     |
| Embeddings   | Ready for similarity search |
| Database     | Schema created              |
| PDFs         | Fully indexed               |
| API          | Ready to accept queries     |

The system is now capable of:

* Retrieving relevant chunks
* Answering questions via RAG
* Falling back to `"I Don't Know"` if context is insufficient (handled later in pipeline)

---

# Detailed DFD Startup Flow Diagram

<img width="10340" height="12126" alt="startup_flow_expanded" src="https://github.com/user-attachments/assets/fd18d413-320d-4eb0-8afd-07dd9cdd7d5b" />

---

# Timeline Visualization

<img width="4604" height="1201" alt="Timeline Visualization" src="https://github.com/user-attachments/assets/1baa6a9f-0cf1-4b21-9a1d-79becc267b6e" />

---

# Chat Endpoint Flow 

## DFD for chat flow

<img width="705" height="3633" alt="chat_flow" src="https://github.com/user-attachments/assets/ab9ecdb9-a77d-4ab1-b174-ef3cc98434bc" />


## 1. System Architecture Overview

### 1.1 High-Level Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│ ChatService │────▶│  Vector     │
│             │◀────│   Router    │◀────│  (RAG)      │◀────│  Store      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                   ┌─────────────┐     ┌─────────────┘     ┌─────────────┘
                   │ PostgreSQL  │     │   LLM (Groq)      │   Document
                   │ Chat History│     │                   │   Embeddings
                   └─────────────┘     └─────────────┘     └─────────────┘
```

### 1.2 Core Components
1. **FastAPI Application** - Web framework handling HTTP requests
2. **ChatService** - Main RAG orchestration logic
3. **Vector Store** - Semantic search and document retrieval
4. **LLM Integration** - Groq API for answer generation
5. **Chat History Manager** - PostgreSQL-based conversation persistence

## 2. Chat Endpoint Workflow

### 2.1 Step-by-Step Flow Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Chat Endpoint                               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Request Validation & Authentication                                 │
│    - Validate ChatRequest schema                                       │
│    - Extract session/authentication from request                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. History Initialization                                              │
│    - Create ChatHistoryManager with DB session                         │
│    - Persist user query to database                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Context Retrieval                                                   │
│    - Fetch last 3 conversation turns (n_turns=3)                       │
│    - Returns: [{role, content}, ...] in chronological order            │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. ChatService Initialization                                          │
│    - Inject vector_store (from app.state)                             │
│    - Inject chat_history_manager                                      │
│    - Initialize LLM (Groq) if API key available                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. RAG Pipeline Execution                                              │
│    (Detailed in Section 3)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. Response Processing                                                 │
│    - Format source documents                                           │
│    - Persist assistant response to DB                                  │
│    - Include: answer, confidence, sources                              │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. Error Handling & Logging                                            │
│    - Structured logging with session context                           │
│    - HTTP 500 for unexpected errors                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Detailed RAG Pipeline (ChatService.get_answer)

### 3.1 Query Analysis & Preprocessing
```python
# Step 1: Query Type Analysis
query_analysis = {
    "word_count": len(words),
    "is_question": "?" in query,
    "query_type": "definition"|"factual"|"explanation"|"comparison"|"general",
    "complexity": "simple"|"medium"|"complex"
}

# Step 2: Follow-up Detection
followup_indicators = [
    "it", "that", "this", "they", "them", "their",
    "also", "what about", "how about", "and", "but"
]
```

### 3.2 Context-Aware Query Rewriting
```
┌─────────────────────────────────────────────────────────────────────────┐
│   Original Query: "What about their pricing?"                          │
│   Context: Previous conversation about "Company X features"            │
│                                                                        │
│   LLM Rewriting Process:                                              │
│   1. System Prompt: "Rewrite query to be standalone..."               │
│   2. Input: Conversation history + New query                          │
│   3. Output: "What is the pricing for Company X?"                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Document Retrieval & Filtering
```python
# Vector Store Query
results = vector_store.query(
    query=effective_query,          # Original or rewritten query
    n_results=settings.SIMILARITY_TOP_K  # Configurable (e.g., 5)
)

# Results structure:
{
    'documents': [[doc1, doc2, ...]],      # 2D array (1 query × n_results)
    'metadatas': [[meta1, meta2, ...]],    # Corresponding metadata
    'distances': [[dist1, dist2, ...]]     # Euclidean/L2 distances
}

# Threshold Filtering
similarity = 1 / (1 + distance)           # Convert distance to similarity
if similarity >= settings.SIMILARITY_THRESHOLD:  # e.g., 0.7
    include_document()
```

### 3.4 Answer Generation Process
```
┌─────────────────────────────────────────────────────────────────────────┐
│   Answer Generation Pipeline                                           │
├─────────────────────────────────────────────────────────────────────────┤
│   Input:                                                              │
│   - Query: "What is machine learning?"                                │
│   - Context: Retrieved documents concatenated                         │
│   - System Prompt: "Answer based ONLY on context..."                  │
│                                                                        │
│   LLM Processing:                                                     │
│   1. Format prompt with context and query                             │
│   2. Generate response with temperature=0.1, max_tokens=512           │
│   3. Validate response doesn't contain "I don't know" phrases         │
│                                                                        │
│   Output: Concise, context-based answer or "I Don't Know"             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Confidence Calculation Algorithm
```python
# Multi-factor Confidence Scoring
factors = {
    "avg_similarity": np.mean(similarity_scores) * weight1,
    "best_score": max(similarity_scores) * weight2,
    "consistency": 1/(1 + 10*variance) * weight3,
    "doc_count": min(len(scores)/3, 1.0) * weight4,
    "clustering": clustering_score * weight5  # Top scores proximity
}

# Threshold Adjustments
if complexity == "complex":
    high_threshold += 0.05
    medium_threshold += 0.05
elif complexity == "simple":
    high_threshold -= 0.05
    medium_threshold -= 0.05
```

## 4. Data Flow & State Management

### 4.1 Chat History Data Model
```sql
-- Message Table Schema
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    role VARCHAR(20) NOT NULL,           -- 'user' or 'assistant'
    content TEXT NOT NULL,
    confidence VARCHAR(10),              -- 'high', 'medium', 'low'
    sources JSONB,                       -- Source documents metadata
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexing for Performance
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX idx_messages_role ON messages(role);
```

### 4.2 Context Retrieval Logic
```python
# Recent Context Flow
1. Query: SELECT role, content FROM messages 
           ORDER BY timestamp DESC LIMIT n_turns * 2
2. Process: Reverse results for chronological order
3. Output: [{"role": "user", "content": "..."}, ...]
```

### 4.3 Source Document Formatting
```python
SourceDocument(
    content=doc[:200] + "...",          # Truncated for response
    source=meta.get('source', 'unknown'),
    page=meta.get('page'),
    similarity_score=round(score, 3)    # 3 decimal places
)
```

## 5. Error Handling & Resilience

### 5.1 Graceful Degradation Paths
```
Primary Path: LLM + RAG → Fallback: Document Snippet → Fallback: "I Don't Know"

1. LLM Failure → Use document snippet
2. Vector Store Failure → Return "I Don't Know"
3. Query Rewriting Failure → Use original query
4. DB Connection Issues → Continue without history
```

### 5.2 Structured Logging
```python
logger.debug(f"""
Session: {session_id} | 
Query analysis: {query_analysis} | 
Scores: {relevant_scores} | 
Context turns: {len(conversation_context)}
""")
```

## 6. Configuration Parameters

### 6.1 Key Settings
```python
# Application Settings
SIMILARITY_THRESHOLD = 0.7      # Minimum similarity score
SIMILARITY_TOP_K = 5            # Number of documents to retrieve
GROQ_MODEL = "mixtral-8x7b-32768"  # LLM model
MAX_HISTORY_TURNS = 3           # Conversation context window

# Confidence Thresholds
confidence_config = {
    "high_threshold": 0.8,
    "medium_threshold": 0.6,
    "distance_to_similarity": 1.5,
    "min_relevant_docs": 1
}
```

Perfect point — this is an **important architectural distinction** and absolutely worth documenting clearly because it directly impacts **hallucination control, scalability, and token efficiency**.

Below is an **add-on section** you can plug into the existing technical document.
It explains **how and why conversation history is maintained**, and **why it is deliberately excluded from answer generation**.

---

# Conversation History Strategy (Critical Design Decision)

## Purpose of Conversation History

The conversation history in this system is **NOT used for answer generation**.

Instead, it serves **one focused and controlled purpose**:

> **To improve retrieval accuracy by rewriting follow-up queries into standalone queries.**

This design ensures that **context is applied only at the retrieval layer**, not at the generation layer.

---

## Where History Is Used (And Where It Is Not)

### ✅ Used For

* **Follow-up detection**
* **Query rewriting**
* **Retrieval context enrichment**

### ❌ Explicitly NOT Used For

* Answer generation
* Prompt conditioning
* Multi-turn reasoning
* Memory-based hallucination

This separation is **intentional and strategic**.

---

## Query Rewriting with Conversation Context

### When Rewriting Happens

Query rewriting is triggered only if:

1. Conversation context exists
2. The query is detected as a **follow-up**
3. LLM is available

```python
if conversation_context and self._is_followup_query(query, conversation_context):
    effective_query = await self._rewrite_query_with_context(query, conversation_context)
```

---

### How Context Is Applied

* Only the **last 6 messages** are used
* Messages are converted into a structured format:

  ```
  User: ...
  Assistant: ...
  ```

```python
context_str = "\n".join([
    f"{msg['role'].capitalize()}: {msg['content']}"
    for msg in conversation_context[-6:]
])
```

This ensures:

* Minimal token usage
* High signal-to-noise ratio
* No historical drift

---

### Rewriting Prompt Strategy

The LLM is **strictly instructed** to:

* Produce a **standalone query**
* Include only **necessary context**
* Avoid verbosity
* Preserve original intent

Key instruction:

> *“If the query is already standalone, return it as is.”*

This avoids unnecessary transformations.

---

### Safety Guards in Rewriting

* Empty output → fallback to original query
* Excessively long output (>500 chars) → fallback
* Any exception → fallback

This guarantees that **query rewriting can never break the pipeline**.

---

## Retrieval Uses Rewritten Query Only

Once rewritten:

```python
results = self.vector_store.query(
    effective_query,
    n_results=settings.SIMILARITY_TOP_K
)
```

Important points:

* The **vector store sees only the rewritten query**
* Conversation history is **never embedded**
* Retrieval remains **stateless and scalable**

---

## Answer Generation Is Context-Strict

When generating the final answer:

```python
answer = await self._generate_groq_answer(
    query=query,
    context=context
)
```

### Key Observations

* **Conversation history is NOT passed**
* Only retrieved documents are used
* The original user query is preserved
* Prevents cross-turn hallucination

This enforces **hard grounding**.

---

## Why This Design Matters (Strategic Rationale)

### 1. Hallucination Control

Passing conversation history to the LLM during answer generation can cause:

* Memory blending
* Implicit assumptions
* Answering beyond documents

By excluding it:

* The model **cannot invent context**
* Answers stay **document-bound**

---

### 2. Retrieval-First Context Handling

This system treats context as a **retrieval problem**, not a **generation problem**.

* Context is resolved **before** the LLM sees anything
* The LLM only answers what retrieval provides

This aligns with **best-in-class RAG architectures**.

---

### 3. Scalability & Cost Efficiency

* History does **not increase generation tokens**
* Query rewriting happens only when needed
* Retrieval remains fast and cache-friendly

This is critical at scale.


---

## Summary of History Strategy

| Layer               | Uses History?           | Why                         |
| ------------------- | ----------------------- | --------------------------- |
| Follow-up detection | ✅                       | Detect contextual queries   |
| Query rewriting     | ✅                       | Improve retrieval relevance |
| Answer generation   | ❌                       | Prevent hallucination       |
| Confidence scoring  | ❌                       | Deterministic scoring       |

---

### Final Takeaway

> **Conversation history is treated as a retrieval aid, not as knowledge.**

---

## This is how the chat is working 

---

## 🔗 API Endpoints – Technical Overview

### 🔹 **POST `/chat`**

**Core conversational inference endpoint**

* Accepts a user query and processes it through the RAG pipeline
* Persists user input to chat history (database-backed)
* Retrieves recent conversation context for follow-up resolution
* Executes vector-based document retrieval and LLM answer generation
* Returns:

  * Knowledge-grounded answer
  * Source documents with page references
  * Confidence score for answer reliability

---

### 🔹 **GET `/chat/history`**

**Conversation state retrieval**

* Fetches stored chat messages from persistent storage
* Returns both user and assistant messages
* Supports pagination via `limit` parameter
* Useful for UI rendering, debugging, and audit logs

---

### 🔹 **DELETE `/chat`**

**Conversation reset endpoint**

* Deletes all stored chat messages
* Resets conversational context
* Does not impact the vector store or indexed documents

---

### 🔹 **GET `/health`**

**System health & readiness probe**

* Validates vector store initialization
* Returns total indexed document chunk count
* Designed for deployment health checks and monitoring systems

---

### 🔹 **GET `/stats`**

**Knowledge base statistics endpoint**

* Returns total number of indexed document chunks
* Exposes active vector collection name
* Indicates operational status of retrieval layer

---

### 🔹 **POST `/reset`**

**Knowledge base administration endpoint**

* Completely clears the vector store
* Deletes all embedded document chunks
* Intended for development or controlled administrative use
* ⚠️ **Destructive and irreversible operation**

---

## 🧩 Endpoint Responsibility Summary

| Endpoint         | Responsibility                                        |
| ---------------- | ----------------------------------------------------- |
| `/chat`          | Context-aware, knowledge-grounded response generation |
| `/chat/history`  | Persistent conversation retrieval                     |
| `/chat` (DELETE) | Conversation state reset                              |
| `/health`        | System readiness & health validation                  |
| `/stats`         | Knowledge base introspection                          |
| `/reset`         | Vector store reset (admin-only)                       |


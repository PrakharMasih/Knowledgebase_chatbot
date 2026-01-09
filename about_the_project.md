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


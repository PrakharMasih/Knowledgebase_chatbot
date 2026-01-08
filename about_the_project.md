# PART 1: Codebase Walkthrough (Current Implementation)

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

### 2. **Data Ingestion & Document Processing**

**File: `document_processor.py`**

The system processes PDF documents through the following pipeline:

1. **PDF Loading**: Uses `PyPDFLoader` from LangChain to extract text from PDFs in the `data/pdfs/` directory
2. **Text Chunking**: Applies `RecursiveCharacterTextSplitter` with:
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Separators: `["\n\n", "\n", " ", ""]`
3. **Enhanced Chunking**: Additional chunking strategies are available (paragraph, sentence, fixed) for better retrieval
4. **Metadata Extraction**: Captures source filename, page number, and chunk index
5. **Unique ID Generation**: Creates MD5 hash-based IDs for each chunk to prevent duplicates

**Process Flow**:
```
PDF File → PyPDFLoader → Pages → RecursiveCharacterTextSplitter → Chunks → Vector Store
```

### 3. **Embedding Generation & Storage**

**Files: `embeddings.py`, `vector_store.py`**

**Embedding Generation**:
- Uses **Sentence Transformers** model: `all-MiniLM-L6-v2`
- Normalized embeddings for better similarity comparison
- Runs on CPU with 384-dimensional vectors

**Vector Storage (ChromaDB)**:
- **Persistent storage** at `vectordb/` directory
- Collection name: `pdf_documents`
- Distance metric: L2 (Euclidean distance)
- Features:
  - Document deduplication via chunk hashing
  - Metadata storage (source, page, chunk_index)
  - Sub-chunking strategy for improved granularity

**Key Methods**:
- `add_documents()`: Stores document chunks with embeddings
- `query()`: Performs vector similarity search
- `document_exists()`: Prevents duplicate ingestion

### 4. **Retrieval Mechanism**

**File: `chat_service.py`**

The retrieval system implements a sophisticated multi-stage approach:

#### **Stage 1: Query Analysis**
- **Query Type Detection**: Identifies factual, definition, explanation, or comparison queries
- **Complexity Assessment**: Categorizes as simple/medium/complex based on word count
- **Follow-up Detection**: Checks if query references previous conversation context

#### **Stage 2: Query Rewriting (Contextual)**
When a follow-up query is detected:
1. Retrieves last 6 conversation messages
2. Uses Groq LLM to rewrite query as standalone
3. Falls back to original if rewriting fails

**Example**:
```
Previous: "What is machine learning?"
Current: "How about deep learning?"
Rewritten: "What is deep learning and how does it differ from machine learning?"
```

#### **Stage 3: Vector Search**
- Queries ChromaDB with effective query (original or rewritten)
- Retrieves top K results (default: 3)
- Returns documents, metadata, and L2 distances

#### **Stage 4: Similarity Filtering**
Converts L2 distance to similarity score:
```python
similarity = 1 / (1 + distance)
```

Filters by dynamic threshold:
- Short queries (≤3 words): 0.7
- Definition queries: 0.7
- Factual queries: 0.8
- Complex/explanation queries: 0.5
- Default: 0.6

### 5. **Confidence Scoring**

**File: `chat_service.py` - `_calculate_confidence()`**

Implements a **weighted multi-factor confidence system**:

**Factors**:
1. **Average Similarity**: Mean of all similarity scores
2. **Best Score**: Highest similarity score
3. **Consistency Score**: `1 / (1 + 10 * variance)` - measures score clustering
4. **Document Count Score**: Normalized by ideal count (3 docs)
5. **Clustering Score**: Measures middle-range spread

**Weighted Formula** (varies by query type):
- Factual queries: Prioritize avg_similarity (40%) and best_score (30%)
- Definition queries: Balance consistency (25%) and avg_similarity (30%)
- General: Balanced weights across all factors

**Thresholds** (adjusted by complexity):
- High confidence: ≥ 0.8
- Medium confidence: ≥ 0.6
- Low confidence: < 0.6

### 6. **Prompting Strategy & Response Generation**

**File: `chat_service.py` - `_generate_groq_answer()`**

**LLM Configuration**:
- Model: Groq API (model specified in settings)
- Temperature: 0.1 (deterministic responses)
- Max tokens: 512

**System Prompt**:
```
You are a helpful assistant that answers questions based ONLY on the provided context.
If conversation history is provided, use it to understand references and follow-up questions.
If the context doesn't contain relevant information, respond with exactly 'I Don't Know'.
Keep answers concise, accurate, and directly address the question.
```

**Prompt Template**:
```
Context:
{retrieved_documents}

Question: {user_query}

Answer:
```

**Fallback Detection**:
If LLM response contains phrases like "don't know", "cannot answer", "no information" → returns "I Don't Know"

### 7. **Chat Flow & Conversation History**

**File: `chat_history.py`, `routes.py`**

**Database Schema** (`models.py`):
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    role VARCHAR NOT NULL,        -- 'user' or 'assistant'
    content TEXT NOT NULL,
    confidence VARCHAR,           -- 'high', 'medium', 'low'
    sources TEXT,                 -- JSON string of sources
    timestamp DATETIME DEFAULT NOW
);
```

**Chat Flow** (per request):
1. User sends query via POST `/api/v1/chat`
2. Store user message in database
3. Retrieve last 3 conversation turns (6 messages)
4. Detect if follow-up query
5. Rewrite query if needed
6. Perform retrieval and generation
7. Store assistant response with metadata
8. Return response to user

**Conversation Context**:
- Maintains last N turns in memory (default: 3 turns = 6 messages)
- Used for query rewriting and reference resolution
- Does NOT pass full history to LLM (only context string for rewriting)

### 8. **API Design**

**File: `routes.py`**

**Endpoints**:

1. **POST `/api/v1/chat`**
   - Request: `{"query": "string"}`
   - Response: `{"answer": "string", "sources": [...], "confidence": "high|medium|low"}`

2. **GET `/api/v1/chat/history`**
   - Query param: `limit` (default: 50)
   - Returns: List of messages with metadata

3. **DELETE `/api/v1/chat`**
   - Clears all conversation history

4. **GET `/api/v1/health`**
   - Returns system status and document count

5. **GET `/api/v1/stats`**
   - Returns collection statistics

6. **POST `/api/v1/reset`**
   - Deletes all documents from vector store

### 9. **Key Design Choices**

1. **Async/Await Pattern**: All database and LLM calls are async for better concurrency
2. **No Session Management**: Single global conversation history (suitable for single-user demo)
3. **Strict Knowledge Grounding**: System prompt explicitly forbids external knowledge
4. **Dynamic Thresholding**: Adapts similarity requirements based on query characteristics
5. **Graceful Degradation**: Falls back to "I Don't Know" on insufficient context
6. **Source Attribution**: Every response includes source documents with page numbers

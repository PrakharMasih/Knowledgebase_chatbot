"""
RAG Chatbot FastAPI Application
Main entry point for the application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config.settings import settings
from app.chatbot.routes import router
from app.config.init_db import init_db
from app.core.vector_store import EnhancedVectorStore
from app.core.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Initializes vector store and loads PDFs on startup
    """
    logger.info("Starting RAG Chatbot Application...")

    try:
        # Initialize vector store
        vector_store = EnhancedVectorStore()
        app.state.vector_store = vector_store

        # Initialize document processor
        doc_processor = DocumentProcessor(vector_store)

        # Initialize DB schema
        await init_db()
        logger.info("Chat history database initialized")

        # Load PDFs from data directory on startup
        logger.info("Loading PDF documents into vector store...")
        num_docs = await doc_processor.load_pdfs_from_directory(settings.PDF_DIRECTORY)
        logger.info(f"Successfully loaded {num_docs} documents into vector store")

        # Store document processor in app state
        app.state.doc_processor = doc_processor

        logger.info("Application startup complete!")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

    yield  # Application is running

    # Cleanup on shutdown
    logger.info("Shutting down application...")
    # Add any cleanup code here if needed


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-based Chatbot API that answers questions based on uploaded PDF documents",
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": settings.VERSION,
        "status": "running"
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
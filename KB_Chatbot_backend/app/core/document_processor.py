"""
Document processing service for loading and chunking PDFs
"""
from pathlib import Path
import logging
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.core.vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF document loading and processing"""

    def __init__(self, vector_store: EnhancedVectorStore):
        """
        Initialize document processor

        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def load_pdfs_from_directory(self, directory: Path) -> int:
        """
        Load all PDFs from a directory into the vector store

        Args:
            directory: Path to directory containing PDFs

        Returns:
            Number of documents loaded
        """
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files")
        total_chunks = 0

        for pdf_file in pdf_files:
            try:
                chunks = await self.process_pdf(pdf_file)
                total_chunks += chunks
                logger.info(f"Processed {pdf_file.name}: {chunks} chunks")

            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")

        logger.info(f"Total chunks loaded: {total_chunks}")
        return total_chunks

    async def process_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks created
        """
        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                logger.warning(f"No content extracted from {pdf_path.name}")
                return 0

            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)

            if not chunks:
                logger.warning(f"No chunks created from {pdf_path.name}")
                return 0

            # Prepare data for vector store
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Create unique ID for chunk
                chunk_id = self._generate_chunk_id(pdf_path.name, i)

                # Skip if already exists
                if self.vector_store.document_exists(chunk_id):
                    continue

                documents.append(chunk.page_content)

                # Extract page number from metadata
                page_num = chunk.metadata.get('page', 0) + 1  # Pages are 0-indexed

                metadatas.append({
                    'source': pdf_path.name,
                    'page': page_num,
                    'chunk_index': i
                })

                ids.append(chunk_id)

            # Add to vector store
            if documents:
                self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            return len(documents)

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {str(e)}")
            raise

    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a document chunk

        Args:
            filename: Name of the source file
            chunk_index: Index of the chunk

        Returns:
            Unique chunk ID
        """
        content = f"{filename}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
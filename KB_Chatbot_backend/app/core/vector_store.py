"""
Enhanced Vector Store Management using ChromaDB with hybrid search, reranking, and advanced features
"""
import re
import hashlib
import logging
import chromadb
import numpy as np
from app.config.settings import settings
from typing import List, Dict, Any, Optional
from app.core.embeddings import get_embedding_function
from chromadb.config import Settings as ChromaSettings


logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """vector database operations with hybrid search and reranking"""

    def __init__(self, distance_metric: str = "l2", enable_hybrid: bool = True):
        """
        Initialize ChromaDB client

        Args:
            distance_metric: Distance metric for similarity ("l2", "cosine", or "ip")
            enable_hybrid: Enable hybrid search features
        """
        try:
            # Initialize persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )

            self.embedding_function = get_embedding_function()

            self.distance_metric = distance_metric
            self.enable_hybrid = enable_hybrid


            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"description": "PDF document embeddings"},
            )

            logger.info(f"Enhanced vector store initialized with collection: {settings.COLLECTION_NAME}")
            logger.info(f"Distance metric: {distance_metric}, Hybrid search: {enable_hybrid}")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced vector store: {str(e)}")
            raise


    def _tokenize_document(self, text: str) -> List[str]:
        """Tokenize document text with cleaning"""
        text = re.sub(r'\s+', ' ', text.lower().strip())

        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'had', 'her', 'was',
            'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old',
            'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'way'
        }
        return [t for t in tokens if t not in stop_words]

    def add_documents(
            self,
            documents: List[str],
            metadatas: List[Dict[str, Any]],
            ids: List[str],
            chunk_strategy: str = "paragraph"
    ) -> None:
        """
        Add documents to the vector store with enhanced chunking

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            chunk_strategy: Strategy for chunking ("paragraph", "sentence", "fixed")
        """
        try:
            # Enhanced chunking for better retrieval
            chunked_docs, chunked_metas, chunked_ids = [], [], []

            for doc, meta, doc_id in zip(documents, metadatas, ids):
                chunks = self._chunk_document(doc, chunk_strategy)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunked_docs.append(chunk)
                    chunked_metas.append({
                        **meta,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                    })
                    chunked_ids.append(chunk_id)

            # Add to collection
            self.collection.add(
                documents=chunked_docs,
                metadatas=chunked_metas,
                ids=chunked_ids
            )


            logger.info(f"Added {len(chunked_docs)} chunks from {len(documents)} documents to vector store")
            logger.info(f"Chunking strategy: {chunk_strategy}")

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def _chunk_document(self, document: str, strategy: str = "paragraph") -> List[str]:
        """Chunk document into smaller pieces for better retrieval"""
        if strategy == "paragraph":
            # Split by paragraphs
            paragraphs = re.split(r'\n\s*\n', document.strip())
            chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        elif strategy == "sentence":
            # Split by sentences
            sentences = re.split(r'[.!?]+', document)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 30]

            # Combine short sentences
            combined = []
            current = ""
            for chunk in chunks:
                if len(current) + len(chunk) < 500:
                    current = f"{current} {chunk}".strip()
                else:
                    if current:
                        combined.append(current)
                    current = chunk
            if current:
                combined.append(current)
            chunks = combined
        else:  # "fixed"
            # Fixed-size chunks
            chunk_size = 500
            overlap = 100
            chunks = []
            start = 0
            while start < len(document):
                end = start + chunk_size
                chunk = document[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start += chunk_size - overlap

        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk) > 50]

    def query(
            self,
            query_text: str,
            n_results: int = None,
            score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced query with multiple search strategies

        Args:
            query_text: Query text
            n_results: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            Dictionary containing documents, metadatas, and distances
        """
        try:
            if n_results is None:
                n_results = settings.SIMILARITY_TOP_K

            results = self._vector_search(query_text, n_results)

            if score_threshold is not None and results.get('distances'):
                filtered_results = self._filter_by_threshold(
                    results, score_threshold, self.distance_metric
                )
                if filtered_results:
                    results = filtered_results
            self._log_query_performance(query=query_text, results=results)
            return results

        except Exception as e:
            logger.error(f"Failed to query vector store: {str(e)}")
            raise

    def _vector_search(
            self,
            query_text: str,
            n_results: int,
    ) -> Dict[str, Any]:
        """Pure vector similarity search"""
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        results = self.collection.query(**query_params)
        return results


    def _get_documents_by_ids(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Retrieve documents by IDs with proper distance calculation"""
        if not doc_ids:
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

        try:
            results = self.collection.get(ids=doc_ids)

            if not results or not results.get('documents'):
                return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

            distances = []

            if len(results['documents']) > 1:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                model = SentenceTransformer('all-MiniLM-L6-v2')

                doc_embeddings = model.encode(results['documents'])

                query_embedding = doc_embeddings[0]
                for i in range(len(doc_embeddings)):
                    if self.distance_metric == "cosine":
                        cos_sim = np.dot(query_embedding, doc_embeddings[i]) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[i])
                        )
                        distance = 1 - cos_sim
                    else:
                        distance = np.linalg.norm(query_embedding - doc_embeddings[i])

                    distances.append(float(distance))
            else:
                distances = [0.5]

            return {
                'documents': [results['documents']],
                'metadatas': [results.get('metadatas', [])],
                'distances': [distances],
                'ids': [results['ids']]
            }

        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {str(e)}")
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[0.5] * len(doc_ids)],
                'ids': [doc_ids]
            }

    @staticmethod
    def _filter_by_threshold(
            results: Dict[str, Any],
            threshold: float,
            distance_metric: str = "l2"
    ) -> Optional[Dict[str, Any]]:
        """Filter results by similarity threshold"""
        if not results.get('distances') or not results['distances'][0]:
            return None

        filtered_docs, filtered_metas, filtered_dists, filtered_ids = [], [], [], []

        for i, distance in enumerate(results['distances'][0]):
            # Convert to similarity based on distance metric
            if distance_metric == "l2":
                similarity = 1 / (1 + distance)  # Convert L2 to similarity
            elif distance_metric == "cosine":
                similarity = 1 - distance  # Cosine distance is already 0-2
            else:  # inner product
                similarity = distance  # Assuming higher is better

            if similarity >= threshold:
                filtered_docs.append(results['documents'][0][i])
                filtered_metas.append(results['metadatas'][0][i])
                filtered_dists.append(distance)
                filtered_ids.append(results['ids'][0][i])

        if not filtered_docs:
            return None

        return {
            'documents': [filtered_docs],
            'metadatas': [filtered_metas],
            'distances': [filtered_dists],
            'ids': [filtered_ids]
        }

    @staticmethod
    def _log_query_performance(query: str, results: Dict):
        """Log query performance metrics"""
        if not results.get('documents') or not results['documents'][0]:
            logger.debug(f"Query: '{query[:50]}...' - No results found ")
            return

        num_results = len(results['documents'][0])
        avg_distance = np.mean(results['distances'][0]) if results['distances'][0] else 0

        logger.info(
            f"Query: '{query[:50]}...' - "
            f"Results: {num_results} - "
            f"Avg Distance: {avg_distance:.3f} - "
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        try:
            count = self.collection.count()

            # Get sample of documents for analysis
            sample = self.collection.get(limit=min(100, count))

            stats = {
                "total_documents": count,
                "embedding_dimension": self.embedding_function.__dict__.get('model', {}).get('max_seq_length', 384),
                "distance_metric": self.distance_metric,
                "sample_analysis": {
                    "avg_document_length": np.mean([len(d) for d in sample['documents']]) if sample['documents'] else 0,
                    "avg_token_count": np.mean([len(self._tokenize_document(d)) for d in sample['documents']]) if sample['documents'] else 0
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}


    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            return 0

    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)"""
        try:
            self.client.delete_collection(settings.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_function
            )

            logger.info("Collection reset successfully")

        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document with given ID exists"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
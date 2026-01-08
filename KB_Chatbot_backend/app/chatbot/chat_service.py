import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.config.settings import settings
from app.core.vector_store import EnhancedVectorStore
from app.core.chat_history import ChatHistoryManager
from app.chatbot.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat queries with RAG and conversation history"""

    def __init__(
            self,
            vector_store: EnhancedVectorStore,
            chat_history_manager: Optional[ChatHistoryManager] = None
    ):
        """
        Initialize chat service with enhanced confidence scoring and history support

        Args:
            vector_store: VectorStore instance
            chat_history_manager: Optional ChatHistoryManager for conversation context
        """
        self.vector_store = vector_store
        self.chat_history_manager = chat_history_manager

        # Initialize confidence thresholds
        self.confidence_config = {
            "high_threshold": 0.8,
            "medium_threshold": 0.6,
            "distance_to_similarity": 1.5,
            "min_relevant_docs": 1,
            "short_query_threshold": 0.7,
            "complex_query_threshold": 0.5,
            "fact_query_threshold": 0.8,
            "definition_query_threshold": 0.7
        }

        if settings.GROQ_API_KEY:
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.1,
                max_tokens=512,
                max_retries=2,
            )
            self.use_llm = True
            logger.info("Chat service initialized with Groq LLM and chat history support")
        else:
            self.llm = None
            self.use_llm = False
            logger.warning("No Groq API key provided, LLM features disabled")

    def _get_dynamic_threshold(self, query: str = None) -> float:
        """
        Get dynamic threshold based on query characteristics

        Args:
            query: User query (optional, for query-based thresholding)

        Returns:
            Dynamic threshold value
        """
        default_threshold = 0.6

        if not query:
            return default_threshold

        query_lower = query.lower().strip()
        words = query_lower.split()

        if len(words) <= 3:
            return self.confidence_config["short_query_threshold"]

        if any(word in query_lower for word in ["what is", "define", "definition", "meaning of"]):
            return self.confidence_config["definition_query_threshold"]

        if any(word in query_lower for word in ["how many", "how much", "when", "where", "who"]):
            return self.confidence_config["fact_query_threshold"]

        if any(word in query_lower for word in ["explain", "describe", "why", "how does"]):
            return self.confidence_config["complex_query_threshold"]

        if "?" in query:
            return 0.7

        return default_threshold

    def _analyze_query_type(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine its type and characteristics

        Args:
            query: User query

        Returns:
            Dictionary with query analysis
        """
        query_lower = query.lower().strip()
        words = query_lower.split()

        analysis = {
            "word_count": len(words),
            "is_question": "?" in query,
            "query_type": "general",
            "complexity": "medium"
        }

        if any(word in query_lower for word in ["what is", "define", "definition"]):
            analysis["query_type"] = "definition"
        elif any(word in query_lower for word in ["how many", "how much", "when", "where", "who"]):
            analysis["query_type"] = "factual"
        elif any(word in query_lower for word in ["explain", "describe", "why", "how does"]):
            analysis["query_type"] = "explanation"
        elif any(word in query_lower for word in ["compare", "difference", "similar"]):
            analysis["query_type"] = "comparison"

        if len(words) <= 3:
            analysis["complexity"] = "simple"
        elif len(words) > 8:
            analysis["complexity"] = "complex"

        return analysis

    def _is_followup_query(
            self,
            query: str,
            conversation_context: List[Dict[str, str]]
    ) -> bool:
        """
        Determine if query is a follow-up question

        Args:
            query: Current user query
            conversation_context: Previous conversation messages

        Returns:
            True if query appears to be a follow-up
        """
        if not conversation_context:
            return False

        query_lower = query.lower().strip()

        # Indicators of follow-up queries
        followup_indicators = [
            "it", "that", "this", "they", "them", "their",
            "also", "additionally", "moreover", "furthermore",
            "what about", "how about", "and", "but",
            "more", "another", "other", "same"
        ]

        words = query_lower.split()
        if len(words) <= 5 and any(indicator in query_lower for indicator in followup_indicators):
            return True

        followup_starts = ["what about", "how about", "and what", "can you also", "what else"]
        if any(query_lower.startswith(pattern) for pattern in followup_starts):
            return True

        return False

    async def _rewrite_query_with_context(
            self,
            query: str,
            conversation_context: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query using conversation context for better retrieval

        Args:
            query: Original user query
            conversation_context: Previous conversation messages

        Returns:
            Rewritten query with context
        """
        if not self.use_llm or not conversation_context:
            return query

        try:
            context_str = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in conversation_context[-6:]
            ])

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a query rewriting assistant. Given a conversation history and a new query, "
                 "rewrite the query to be standalone and include necessary context from the conversation. "
                 "If the query is already standalone, return it as is. "
                 "Keep the rewritten query concise and focused on retrieving relevant information."),
                ("human",
                 "Conversation history:\n{context}\n\n"
                 "New query: {query}\n\n"
                 "Rewritten standalone query:")
            ])

            chain = prompt | self.llm
            result = await chain.ainvoke({
                "context": context_str,
                "query": query
            })

            rewritten_query = result.content.strip()

            if len(rewritten_query) > 0 and len(rewritten_query) < 500:
                logger.debug(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                return rewritten_query
            else:
                logger.warning("Query rewriting produced invalid result, using original")
                return query

        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            return query

    def _filter_by_threshold(
            self,
            documents: List[str],
            metadatas: List[dict],
            distances: List[float]
    ) -> Tuple[List[str], List[dict], List[float]]:
        """Filter documents by similarity threshold"""
        filtered_docs = []
        filtered_metadata = []
        filtered_scores = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1 / (1 + dist)

            if similarity >= settings.SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)
                filtered_scores.append(similarity)

        return filtered_docs, filtered_metadata, filtered_scores

    def _calculate_confidence(
            self,
            similarity_scores: List[float],
            query: str = None
    ) -> str:
        """
        Enhanced confidence calculation with multiple factors

        Args:
            similarity_scores: List of similarity scores (0-1)
            query: Original query for context

        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not similarity_scores:
            return "low"

        query_analysis = self._analyze_query_type(query) if query else {}
        query_type = query_analysis.get("query_type", "general")
        complexity = query_analysis.get("complexity", "medium")

        avg_similarity = np.mean(similarity_scores)

        if len(similarity_scores) > 1:
            score_variance = np.var(similarity_scores)
            consistency_score = 1 / (1 + (10 * score_variance))
        else:
            consistency_score = 0.5

        doc_count = len(similarity_scores)
        doc_count_score = min(doc_count / 3, 1.0)

        best_score = max(similarity_scores)

        if len(similarity_scores) >= 3:
            sorted_scores = sorted(similarity_scores)
            middle_range = sorted_scores[-2] - sorted_scores[1] if len(sorted_scores) > 2 else 0
            clustering_score = 1 - (middle_range * 2)
            clustering_score = max(0, min(1, clustering_score))
        else:
            clustering_score = 0.5

        if query_type == "factual":
            weights = {
                "avg_similarity": 0.4,
                "best_score": 0.3,
                "consistency": 0.15,
                "doc_count": 0.1,
                "clustering": 0.05
            }
        elif query_type == "definition":
            weights = {
                "avg_similarity": 0.3,
                "best_score": 0.2,
                "consistency": 0.25,
                "doc_count": 0.15,
                "clustering": 0.1
            }
        else:
            weights = {
                "avg_similarity": 0.35,
                "best_score": 0.25,
                "consistency": 0.2,
                "doc_count": 0.1,
                "clustering": 0.1
            }

        confidence_score = (
                weights["avg_similarity"] * avg_similarity +
                weights["best_score"] * best_score +
                weights["consistency"] * consistency_score +
                weights["doc_count"] * doc_count_score +
                weights["clustering"] * clustering_score
        )

        high_threshold = self.confidence_config["high_threshold"]
        medium_threshold = self.confidence_config["medium_threshold"]

        if complexity == "complex":
            high_threshold += 0.05
            medium_threshold += 0.05
        elif complexity == "simple":
            high_threshold -= 0.05
            medium_threshold -= 0.05

        if confidence_score >= high_threshold:
            return "high"
        elif confidence_score >= medium_threshold:
            return "medium"
        else:
            return "low"

    async def get_answer(
            self,
            query: str,
            session_id: Optional[str] = None,
            conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """
        Get answer for a user query using RAG with conversation history support

        Args:
            query: User query
            session_id: Optional session identifier for tracking
            conversation_context: Optional conversation history for context

        Returns:
            ChatResponse with answer and sources
        """
        try:
            effective_query = query
            if conversation_context and self._is_followup_query(query, conversation_context):

                logger.info(f"Detected follow-up query, rewriting with context")
                effective_query = await self._rewrite_query_with_context(query, conversation_context)

            results = self.vector_store.query(
                effective_query,
                n_results=settings.SIMILARITY_TOP_K
            )


            if not results['documents'] or not results['documents'][0]:
                return ChatResponse(
                    answer="I Don't Know",
                    sources=[],
                    confidence="low"
                )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            relevant_docs, relevant_metadata, relevant_scores = self._filter_by_threshold(
                documents, metadatas, distances
            )

            if not relevant_docs:
                return ChatResponse(
                    answer="I Don't Know",
                    sources=[],
                    confidence="low"
                )

            query_analysis = self._analyze_query_type(query)
            logger.debug(
                f"Session: {session_id} | Query analysis: {query_analysis} | "
                f"Scores: {relevant_scores} | Context turns: {len(conversation_context) if conversation_context else 0}"
            )

            context = self._create_context(relevant_docs)

            if self.use_llm:
                # Include conversation context in answer generation
                answer = await self._generate_groq_answer(
                    query=query,
                    context=context
                )
            else:
                answer = self._generate_fallback_answer(relevant_docs)

            source_docs = self._prepare_sources(
                relevant_docs,
                relevant_metadata,
                relevant_scores
            )

            confidence = self._calculate_confidence(relevant_scores, query)

            return ChatResponse(
                answer=answer,
                sources=source_docs,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    def _create_context(self, documents: List[str]) -> str:
        """Create context string from documents"""
        return "\n\n".join([
            f"Document {i + 1}:\n{doc}"
            for i, doc in enumerate(documents)
        ])

    async def _generate_groq_answer(
            self,
            query: str,
            context: str
    ) -> str:
        """
        Generate answer using LangChain's ChatGroq with conversation awareness

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            Generated answer
        """
        try:

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant that answers questions based ONLY on the provided context. "
                 "If conversation history is provided, use it to understand references and follow-up questions. "
                 "If the context doesn't contain relevant information to answer the question, "
                 "respond with exactly 'I Don't Know'. "
                 "Keep answers concise, accurate, and directly address the question."),
                ("human",
                 "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
            ])

            chain = prompt | self.llm
            ai_msg = await chain.ainvoke({
                "context": context,
                "query": query
            })

            answer = ai_msg.content.strip()

            if any(phrase in answer.lower() for phrase in [
                "don't know", "do not know", "cannot answer",
                "no information", "not sure", "unable to"
            ]):
                return "I Don't Know"

            return answer

        except Exception as e:
            logger.error(f"LangChain Groq generation failed: {str(e)}")
            return "I Don't Know"

    def _generate_fallback_answer(self, documents: List[str]) -> str:
        """Generate a simple answer when LLM is not available"""
        if documents:
            return f"Based on the documents: {documents[0][:300]}..."
        return "I Don't Know"

    def _prepare_sources(
            self,
            documents: List[str],
            metadatas: List[dict],
            scores: List[float]
    ) -> List[SourceDocument]:
        """Prepare source documents for response"""
        sources = []

        for doc, meta, score in zip(documents, metadatas, scores):
            sources.append(SourceDocument(
                content=doc[:200] + "..." if len(doc) > 200 else doc,
                source=meta.get('source', 'unknown'),
                page=meta.get('page'),
                similarity_score=round(score, 3)
            ))

        return sources
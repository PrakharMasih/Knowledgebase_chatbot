"""
API route definitions
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from app.core.chat_history import ChatHistoryManager
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.chatbot.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse
)
from app.chatbot.chat_service import ChatService
from app.config.database import get_db


logger = logging.getLogger(__name__)
router = APIRouter()




@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        vector_store = req.app.state.vector_store
        history = ChatHistoryManager(db)

        await history.add_message(
            role="user",
            content=request.query,
        )

        conversation_context = await history.get_recent_context(n_turns=3)

        chat_service = ChatService(
            vector_store=vector_store,
            chat_history_manager=history,
        )

        response = await chat_service.get_answer(
            query=request.query,
            conversation_context=conversation_context,
        )

        sources_data = [
            {
                "content": src.content,
                "source": src.source,
                "page": src.page,
                "similarity_score": src.similarity_score,
            }
            for src in response.sources
        ]

        await history.add_message(
            role="assistant",
            content=response.answer,
            confidence=response.confidence,
            sources=sources_data,
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/chat/history")
async def get_chat_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    try:
        history = ChatHistoryManager(db)
        messages = await history.get_chat_history(limit=limit)

        return {
            "message_count": len(messages),
            "messages": messages,
        }

    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@router.delete("/chat")
async def delete_chat(db: AsyncSession = Depends(get_db)):
    try:
        history = ChatHistoryManager(db)
        await history.delete_all_messages()

        return {
            "message": "Chat deleted successfully",
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete chat")


@router.get("/health", response_model=HealthResponse)
async def detailed_health_check(req: Request):
    """
    Detailed health check endpoint

    Args:
        req: FastAPI Request object

    Returns:
        HealthResponse with system status
    """
    try:
        vector_store = req.app.state.vector_store
        doc_count = vector_store.get_collection_count()

        return HealthResponse(
            status="healthy",
            vector_store="initialized",
            documents_count=doc_count
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            vector_store="error",
            documents_count=0
        )


@router.get("/stats")
async def get_stats(req: Request):
    """
    Get statistics about the knowledge base

    Args:
        req: FastAPI Request object

    Returns:
        Dictionary with statistics
    """
    try:
        vector_store = req.app.state.vector_store
        doc_count = vector_store.get_collection_count()

        return {
            "total_chunks": doc_count,
            "collection_name": vector_store.collection.name,
            "status": "operational"
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@router.post("/reset")
async def reset_knowledge_base(req: Request):
    """
    Reset the knowledge base (delete all documents)
    WARNING: This will delete all stored documents

    Args:
        req: FastAPI Request object

    Returns:
        Success message
    """
    try:
        vector_store = req.app.state.vector_store
        vector_store.reset_collection()

        logger.warning("Knowledge base has been reset")

        return {
            "message": "Knowledge base reset successfully",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error resetting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset knowledge base: {str(e)}"
        )
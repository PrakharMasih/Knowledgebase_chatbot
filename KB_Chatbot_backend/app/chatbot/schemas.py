"""
Pydantic models for API schemas with chat history support
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User query", min_length=1)



class SourceDocument(BaseModel):
    """Source document with metadata"""
    content: str = Field(..., description="Document content snippet")
    source: str = Field(..., description="Source file or URL")
    page: Optional[int] = Field(None, description="Page number if applicable")
    similarity_score: float = Field(..., description="Relevance score")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used") #type:ignore
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")



class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Overall system status")
    vector_store: str = Field(..., description="Vector store status")
    documents_count: int = Field(..., description="Number of documents in vector store")
    chat_history: Optional[str] = Field(None, description="Chat history database status")


class ChatMessage(BaseModel):
    """Individual chat message model"""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    confidence: Optional[str] = Field(None, description="Confidence level for assistant messages")
    sources: Optional[List[SourceDocument]] = Field(None, description="Sources for assistant messages")
    timestamp: str = Field(..., description="Message timestamp")


class SessionInfo(BaseModel):
    """Chat session information"""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages in session")
    metadata: Optional[dict] = Field(None, description="Additional session metadata")

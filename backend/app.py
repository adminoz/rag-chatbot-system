import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from config import config
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('rag_system.log')  # File output
    ]
)

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Create logger for this module
logger = logging.getLogger(__name__)

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    """Model for source information with optional links"""
    text: str
    link: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[SourceInfo]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        logger.info(f"Processing query request - Query: '{request.query[:100]}...' - Session: {request.session_id}")
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        # Convert sources to SourceInfo objects
        source_info_list = []
        for source in sources:
            if isinstance(source, dict):
                source_info_list.append(SourceInfo(
                    text=source.get('text', ''),
                    link=source.get('link')
                ))
            else:
                # Handle legacy string sources
                source_info_list.append(SourceInfo(text=str(source), link=None))
        
        return QueryResponse(
            answer=answer,
            sources=source_info_list,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    logger.info("Starting RAG System application...")
    # Try relative path first (when running from backend/), then absolute path (when running from root)
    docs_paths = ["../docs", "docs"]
    
    for docs_path in docs_paths:
        if os.path.exists(docs_path):
            logger.info(f"Loading initial documents from {docs_path}...")
            try:
                courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
                logger.info(f"Successfully loaded {courses} courses with {chunks} chunks")
                break
            except Exception as e:
                logger.error(f"Error loading documents from {docs_path}: {e}")
    else:
        logger.warning("No docs folder found in expected locations")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
# Try relative path first (when running from backend/), then absolute path (when running from root)
frontend_paths = ["../frontend", "frontend"]
frontend_dir = None

for path in frontend_paths:
    if os.path.exists(path):
        frontend_dir = path
        break

if frontend_dir:
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
else:
    print("Warning: Frontend directory not found in expected locations")
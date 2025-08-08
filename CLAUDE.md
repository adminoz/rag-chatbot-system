# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (create .env file)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application
```bash
# Quick start using provided script
chmod +x run.sh
./run.sh

# Manual start (from backend directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials with the following architecture:

```
┌─────────────────┐    ┌──────────────────────────────────────────┐
│   Frontend      │    │              Backend                     │
│                 │    │                                          │
│  ┌───────────┐  │    │  ┌─────────────┐   ┌─────────────────┐  │
│  │index.html │  │    │  │   app.py    │   │  rag_system.py  │  │
│  │script.js  │  │◄──►│  │(FastAPI)    │◄─►│  (Orchestrator) │  │
│  │style.css  │  │    │  └─────────────┘   └─────────────────┘  │
│  └───────────┘  │    │         │                   │           │
└─────────────────┘    │         │                   │           │
                       │         ▼                   ▼           │
┌─────────────────┐    │  ┌─────────────┐   ┌─────────────────┐  │
│     docs/       │    │  │Static Files │   │Document Processor│  │
│                 │────┼─►│   Handler   │   │                 │  │
│course1_script.txt│   │  └─────────────┘   └─────────────────┘  │
│course2_script.txt│   │                             │           │
│course3_script.txt│   │                             ▼           │
│course4_script.txt│   │  ┌─────────────┐   ┌─────────────────┐  │
└─────────────────┘    │  │AI Generator │◄──┤  Vector Store   │  │
                       │  │(Claude API) │   │   (ChromaDB)    │  │
                       │  └─────────────┘   └─────────────────┘  │
                       │         ▲                   ▲           │
                       │         │                   │           │
                       │  ┌─────────────┐   ┌─────────────────┐  │
                       │  │Search Tools │   │Session Manager  │  │
                       │  │             │   │                 │  │
                       │  └─────────────┘   └─────────────────┘  │
                       └──────────────────────────────────────────┘
```

### Core Components

#### **1. FastAPI Backend** (`backend/app.py`)
- Main web server with CORS middleware and trusted host configuration
- Serves static frontend files with no-cache headers for development
- Exposes REST API endpoints: `/api/query` and `/api/courses`
- Loads initial documents on startup from `docs/` folder

#### **2. RAG System Orchestrator** (`backend/rag_system.py`)
- Central coordinator managing all system components
- Handles document ingestion and query processing workflow
- Implements tool-based search approach for AI function calling
- Manages conversation context and response generation

#### **3. Vector Store** (`backend/vector_store.py`)
- ChromaDB-based persistent vector storage
- Separate collections for course metadata and content chunks
- Semantic search using sentence-transformers embeddings
- Returns structured `SearchResults` with documents, metadata, and distances

#### **4. AI Generator** (`backend/ai_generator.py`)
- Anthropic Claude API integration with function calling
- Specialized system prompt for educational content
- Tool-aware response generation with search capabilities
- Conversation history support for multi-turn dialogs

#### **5. Document Processor** (`backend/document_processor.py`)
- Processes PDF, DOCX, and TXT files from course materials
- Extracts course metadata (title, instructor, lessons)
- Creates text chunks with configurable size and overlap
- Generates `Course` and `CourseChunk` models for storage

#### **6. Session Manager** (`backend/session_manager.py`)
- Manages conversation history for multi-turn interactions
- Creates unique session IDs for user conversations
- Maintains conversation context with configurable history limits

#### **7. Search Tools** (`backend/search_tools.py`)
- Tool-based search system for AI function calling
- `CourseSearchTool` provides semantic search capabilities
- `ToolManager` handles tool registration and execution
- Returns sources for response attribution

### Data Flow & Processing Pipeline

1. **Document Ingestion**: Course documents in `docs/` → Document Processor → Vector Store
2. **Query Processing**: User Query → RAG System → AI Generator (with tools) → Response
3. **Search Execution**: AI Generator → Search Tools → Vector Store → Retrieved Context
4. **Response Generation**: Retrieved Context + Query + History → Claude API → Final Response
5. **Session Management**: Query/Response pairs stored in Session Manager for context

### Data Models (`backend/models.py`)
- **`Course`**: Represents complete courses with metadata and lessons
- **`Lesson`**: Individual lessons within courses with titles and links  
- **`CourseChunk`**: Text chunks for vector storage with course attribution

### Configuration
- Main config in `backend/config.py` loads from `.env` file
- Uses sentence-transformers model "all-MiniLM-L6-v2" for embeddings
- Claude Sonnet 4 model for response generation
- ChromaDB stores data in `./backend/chroma_db`
- Logging configured in `backend/app.py` with console output and file logging to `backend/rag_system.log`

### Key Models
- `Course`, `Lesson`, `CourseChunk` defined in `backend/models.py`
- Document chunking with 800 character size and 100 character overlap
- Maximum 5 search results and 2 conversation messages in history

### Dependencies
- Python 3.13+ with uv package manager
- FastAPI for web framework
- ChromaDB for vector storage
- Anthropic API for AI generation
- Sentence Transformers for embeddings

## Development Guidelines
- Always use uv to run the server do not use pip directly
- Make sure to use uv to manage all dependencies
- Use uv to run python files instead of python directly

## Memorized Notes
- Do not run run.sh automatically. Execute it manually
- Always run test from root directory not to have dependecy problems
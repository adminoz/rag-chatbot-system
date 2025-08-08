import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from typing import List
from fastapi.testclient import TestClient

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from config import Config
from fastapi import FastAPI


@pytest.fixture
def sample_courses() -> List[Course]:
    """Create sample course data for testing"""
    return [
        Course(
            title="Introduction to Machine Learning",
            instructor="Dr. Smith",
            course_link="https://example.com/ml-course",
            lessons=[
                Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/ml-lesson1"),
                Lesson(lesson_number=2, title="Linear Regression", lesson_link="https://example.com/ml-lesson2"),
                Lesson(lesson_number=3, title="Classification", lesson_link="https://example.com/ml-lesson3")
            ]
        ),
        Course(
            title="Advanced Python Programming",
            instructor="Dr. Johnson", 
            course_link="https://example.com/python-course",
            lessons=[
                Lesson(lesson_number=1, title="Decorators", lesson_link="https://example.com/python-lesson1"),
                Lesson(lesson_number=2, title="Context Managers", lesson_link="https://example.com/python-lesson2")
            ]
        )
    ]


@pytest.fixture
def sample_course_chunks() -> List[CourseChunk]:
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a method of data analysis that automates analytical model building.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Linear regression is a linear approach to modeling relationships between variables.",
            course_title="Introduction to Machine Learning", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Python decorators are a way to modify or extend functions without permanently modifying them.",
            course_title="Advanced Python Programming",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Context managers allow you to allocate and release resources precisely when you want to.",
            course_title="Advanced Python Programming",
            lesson_number=2, 
            chunk_index=1
        )
    ]


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_db_path):
    """Create test configuration with temporary database"""
    config = Config()
    config.CHROMA_PATH = temp_db_path
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"  # Mock key for testing
    return config


@pytest.fixture
def vector_store(test_config):
    """Create vector store instance for testing"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )


@pytest.fixture
def populated_vector_store(vector_store, sample_courses, sample_course_chunks):
    """Create vector store populated with sample data"""
    # Add course metadata
    for course in sample_courses:
        vector_store.add_course_metadata(course)
    
    # Add course content
    vector_store.add_course_content(sample_course_chunks)
    
    return vector_store


@pytest.fixture
def course_search_tool(populated_vector_store):
    """Create CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)


@pytest.fixture 
def tool_manager(course_search_tool):
    """Create ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generator testing"""
    with patch('anthropic.Anthropic') as mock_client:
        # Mock response object
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        
        mock_client.return_value.messages.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def ai_generator(test_config, mock_anthropic_client):
    """Create AI generator with mocked client"""
    return AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)


class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_search_results_valid(results):
        """Assert that search results are properly formatted"""
        assert hasattr(results, 'documents')
        assert hasattr(results, 'metadata')
        assert hasattr(results, 'distances')
        assert len(results.documents) == len(results.metadata)
        assert len(results.documents) == len(results.distances)
    
    @staticmethod
    def create_mock_tool_response(tool_name: str, tool_input: dict, result: str):
        """Create mock tool use response for testing"""
        mock_content = Mock()
        mock_content.type = "tool_use"
        mock_content.name = tool_name
        mock_content.input = tool_input
        mock_content.id = "test_tool_id"
        
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "tool_use"
        
        return mock_response, result


@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils


@pytest.fixture
def test_client():
    """Create FastAPI test client with mocked dependencies to avoid static file issues"""
    with patch('app.RAGSystem') as mock_rag_system_class, \
         patch('app.StaticFiles') as mock_static_files, \
         patch('app.rag_system') as mock_rag_system_instance, \
         patch('app.frontend_dir', None):  # Mock frontend_dir to None to avoid mounting
        
        # Mock the RAG system class
        mock_rag_system = Mock()
        mock_rag_system_class.return_value = mock_rag_system
        
        # Mock the global rag_system instance with comprehensive responses
        mock_rag_system_instance.query.return_value = ("Test answer", [])
        mock_rag_system_instance.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        # Mock session manager
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "test_session_123"
        mock_rag_system_instance.session_manager = mock_session_manager
        
        # Mock StaticFiles to avoid directory path issues in tests
        mock_static_files.return_value = Mock()
        
        # Import app after mocking to ensure mocks are applied
        from app import app
        
        # Create test client
        client = TestClient(app)
        client.mock_rag_system = mock_rag_system_instance
        
        yield client


@pytest.fixture
def mock_query_response():
    """Sample query response for testing"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence.",
        "sources": [
            {"text": "Course 1 - Introduction to ML", "link": "https://example.com/ml-intro"},
            {"text": "Course 2 - ML Fundamentals", "link": "https://example.com/ml-fundamentals"}
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_course_analytics():
    """Sample course analytics for testing"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Introduction to Machine Learning",
            "Advanced Python Programming", 
            "Data Science Fundamentals"
        ]
    }
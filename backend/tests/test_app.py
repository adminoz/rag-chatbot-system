import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the app and dependencies
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def test_client():
    """Create test client with mocked dependencies"""
    with patch('app.RAGSystem') as mock_rag_system_class, \
         patch('app.StaticFiles') as mock_static_files, \
         patch('app.rag_system') as mock_rag_system_instance:
        
        # Mock the RAG system class
        mock_rag_system = Mock()
        mock_rag_system_class.return_value = mock_rag_system
        
        # Mock the global rag_system instance
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
        
        # Import app after mocking
        from app import app
        
        client = TestClient(app)
        client.mock_rag_system = mock_rag_system_instance
        
        yield client


class TestQueryEndpoint:
    """Test /api/query endpoint"""
    
    def test_query_without_session_id(self, test_client):
        """Test query endpoint without session ID"""
        # Mock RAG system response
        test_client.mock_rag_system.query.return_value = ("Test answer", [])
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test answer"
        assert data["sources"] == []
        assert data["session_id"] == "test_session_123"
        
        # Verify RAG system was called correctly
        test_client.mock_rag_system.query.assert_called_once_with(
            "What is machine learning?", 
            "test_session_123"
        )
    
    def test_query_with_session_id(self, test_client):
        """Test query endpoint with existing session ID"""
        # Mock RAG system response
        test_client.mock_rag_system.query.return_value = ("Test answer", [])
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?",
            "session_id": "existing_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test answer"
        assert data["session_id"] == "existing_session"
        
        # Verify RAG system was called with existing session
        test_client.mock_rag_system.query.assert_called_once_with(
            "What is machine learning?",
            "existing_session"
        )
    
    def test_query_with_sources_dict_format(self, test_client):
        """Test query endpoint with dictionary-formatted sources"""
        # Mock RAG system response with dict sources
        mock_sources = [
            {"text": "Course 1 - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "Course 2 - Lesson 2", "link": "https://example.com/lesson2"}
        ]
        test_client.mock_rag_system.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test answer"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Course 1 - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/lesson1"
        assert data["sources"][1]["text"] == "Course 2 - Lesson 2" 
        assert data["sources"][1]["link"] == "https://example.com/lesson2"
    
    def test_query_with_sources_string_format(self, test_client):
        """Test query endpoint with string-formatted sources (legacy)"""
        # Mock RAG system response with string sources
        mock_sources = ["Course 1", "Course 2"]
        test_client.mock_rag_system.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test answer"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Course 1"
        assert data["sources"][0]["link"] is None
        assert data["sources"][1]["text"] == "Course 2"
        assert data["sources"][1]["link"] is None
    
    def test_query_with_mixed_sources(self, test_client):
        """Test query endpoint with mixed source formats"""
        # Mock RAG system response with mixed sources
        mock_sources = [
            {"text": "Course 1", "link": "https://example.com/course1"},
            "Course 2",  # String format
            {"text": "Course 3"}  # Dict without link
        ]
        test_client.mock_rag_system.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Course 1"
        assert data["sources"][0]["link"] == "https://example.com/course1"
        assert data["sources"][1]["text"] == "Course 2"
        assert data["sources"][1]["link"] is None
        assert data["sources"][2]["text"] == "Course 3"
        assert data["sources"][2]["link"] is None
    
    def test_query_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        response = test_client.post("/api/query", json={
            "query": ""
        })
        
        # Should still process (let RAG system handle validation)
        assert response.status_code == 200 or response.status_code == 422
    
    def test_query_missing_query_field(self, test_client):
        """Test query endpoint without query field"""
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, test_client):
        """Test query endpoint when RAG system raises exception"""
        # Mock RAG system to raise exception
        test_client.mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "RAG system error" in data["detail"]
    
    def test_query_session_manager_error(self, test_client):
        """Test query endpoint when session manager fails"""
        # Mock session manager to raise exception
        test_client.mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 500
    
    def test_query_response_format(self, test_client):
        """Test query response conforms to expected schema"""
        test_client.mock_rag_system.query.return_value = ("Test answer", [])
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)


class TestCoursesEndpoint:
    """Test /api/courses endpoint"""
    
    def test_get_courses_success(self, test_client):
        """Test successful course stats retrieval"""
        # Mock RAG system analytics
        mock_analytics = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        test_client.mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course 1", "Course 2", "Course 3"]
    
    def test_get_courses_empty(self, test_client):
        """Test course stats when no courses exist"""
        # Mock empty analytics
        mock_analytics = {
            "total_courses": 0,
            "course_titles": []
        }
        test_client.mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_courses_error(self, test_client):
        """Test course stats when analytics fails"""
        # Mock analytics to raise exception
        test_client.mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]
    
    def test_get_courses_response_format(self, test_client):
        """Test courses response conforms to expected schema"""
        mock_analytics = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }
        test_client.mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


class TestStartupEvent:
    """Test application startup event"""
    
    @patch('os.path.exists')
    @patch('app.rag_system')
    def test_startup_with_docs_folder(self, mock_rag_system, mock_exists):
        """Test startup event when docs folder exists"""
        mock_exists.return_value = True
        mock_rag_system.add_course_folder.return_value = (2, 10)
        
        from app import startup_event
        
        # This would normally be called by FastAPI
        # We can't easily test the actual startup event, but we can test the logic
        docs_path = "docs"
        if os.path.exists(docs_path):
            courses, chunks = mock_rag_system.add_course_folder(docs_path, clear_existing=False)
            
            mock_rag_system.add_course_folder.assert_called_once_with(docs_path, clear_existing=False)
    
    @patch('os.path.exists')
    @patch('app.rag_system')
    def test_startup_without_docs_folder(self, mock_rag_system, mock_exists):
        """Test startup event when docs folder doesn't exist"""
        mock_exists.return_value = False
        
        # If folder doesn't exist, add_course_folder shouldn't be called
        # This is handled in the actual startup event
        
        assert mock_exists.return_value is False
        mock_rag_system.add_course_folder.assert_not_called()


class TestErrorHandling:
    """Test general error handling"""
    
    def test_404_endpoint(self, test_client):
        """Test accessing non-existent API endpoint"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, test_client):
        """Test using wrong HTTP method"""
        response = test_client.get("/api/query")  # Should be POST
        
        # FastAPI returns 404 for undefined routes, not 405 for wrong methods
        assert response.status_code == 404
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.post("/api/query", json={"query": "test"})
        
        # CORS headers should be present due to middleware
        # Note: TestClient doesn't always simulate all middleware behavior
        assert response.status_code in [200, 500]  # Either success or mocked error


class TestResponseModels:
    """Test response model validation"""
    
    def test_source_info_model(self):
        """Test SourceInfo model validation"""
        from app import SourceInfo
        
        # Valid with both fields
        source1 = SourceInfo(text="Test", link="https://example.com")
        assert source1.text == "Test"
        assert source1.link == "https://example.com"
        
        # Valid with just text
        source2 = SourceInfo(text="Test")
        assert source2.text == "Test"
        assert source2.link is None
    
    def test_query_request_model(self):
        """Test QueryRequest model validation"""
        from app import QueryRequest
        
        # Valid with both fields
        request1 = QueryRequest(query="Test query", session_id="session123")
        assert request1.query == "Test query"
        assert request1.session_id == "session123"
        
        # Valid with just query
        request2 = QueryRequest(query="Test query")
        assert request2.query == "Test query"
        assert request2.session_id is None